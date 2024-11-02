# mypy: allow-untyped-defs
from typing import cast, List, Optional

import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
)

__all__ = ["SGDO", "sgdo"]


class SGDO(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        overshoot: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            overshoot=overshoot,
            dampening=dampening,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if overshoot > 0 and (momentum <= 0 or dampening != 0):
            raise ValueError("Overshoot momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True

            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                overshoot=group["overshoot"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss
        
    # TODO: This is only experimental!
    def move_to_base(self):
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []
            self._init_group(group, params, grads, momentum_buffer_list)
            for i, param in enumerate(group["params"]):
                if len(self.state[param]) == 0:
                    return
                if param.grad is None:
                    continue
                param.add_(momentum_buffer_list[i], alpha=group["lr"] * group["overshoot"])
                
    def move_to_overshoot(self):
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []
            self._init_group(group, params, grads, momentum_buffer_list)
            for i, param in enumerate(group["params"]):
                if len(self.state[param]) == 0:
                    return
                if param.grad is None:
                    continue
                param.add_(momentum_buffer_list[i], alpha=-group["lr"] * group["overshoot"])



SGDO.__doc__ = (
    r"""Implements stochastic gradient descent (optionally with overshoot momentum)."""
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        overshoot (float, optional): overshoot factor (default: 0)
        {_maximize_doc}
        {_foreach_doc}
        {_differentiable_doc}
        {_fused_doc}
    """
)


def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    overshoot: float,
    lr: float,
    dampening: float,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        overshoot=overshoot,
        lr=lr,
        dampening=dampening,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


# Done. Note that the only change is under commnted grad = buf
def _single_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    overshoot: float,
    lr: float,
    dampening: float,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if overshoot:
                grad.mul_(overshoot / momentum).add_(buf, alpha=1 + overshoot - (overshoot / momentum))
            else:
                grad = buf

        param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    overshoot: float,
    lr: float,
    dampening: float,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True  # type: ignore[list-item]
    )

    for (
        device_params_,
        device_grads_,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_grads_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs: List[Tensor] = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]
                        ] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = cast(Tensor, device_momentum_buffer_list[i])
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if overshoot:
                torch._foreach_mul_(device_grads, overshoot / momentum)
                torch._foreach_add_(device_grads, bufs, alpha=1 + overshoot - (overshoot / momentum))
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch._utils.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)




# TODO: Fused in not supported
def _fused_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    overshoot: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    no_momentum_buffer = momentum == 0
    is_first_step = (
        all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    )
    if is_first_step:
        for i, g in enumerate(grads):
            momentum_buffer_list[i] = torch.empty_like(g)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=False  # type: ignore[list-item]
    )
    for (device, _), (
        (device_params, device_grads, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device))
        torch._fused_sgd_(
            device_params,
            device_grads,
            [] if no_momentum_buffer else device_momentum_buffer_list,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=is_first_step,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
