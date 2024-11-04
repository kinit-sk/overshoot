import numpy as np
import matplotlib.pyplot as plt

# Define the range for x, avoiding x = 1 to prevent division by zero for f3
x = np.linspace(0, 0.99, 200)

# Define the functions
f1 = np.zeros_like(x)               # f1(x) = 0
f2 = x                              # f2(x) = x
f3 = -x / (x - 1)                   # f3(x) = -x / (x - 1)
f4 = 5.56789422e-09 * np.exp(2.19141010e+01 * x)            # f(x) = e^(17.52 * x)

# Plot the functions with a logarithmic scale on the y-axis
plt.figure(figsize=(6, 4))
plt.subplots_adjust(top=0.99, bottom=0.13) 
plt.plot(x, f1, label="Classical momentum")
plt.plot(x, f2, label="Nesterov's momentum")
plt.plot(x, f3, label="SGD without momentum")
# plt.plot(x, f4, label="Estimated lower bound optimal setting*", linestyle="--",)
plt.plot(x, f4, label="Lower bound optimal setting estimate*", linestyle="--")


# Fill the area above f3
inf = 100000
plt.fill_between(x, f3, inf,  facecolor='none', edgecolor='gray', hatch='//', alpha=0.4, label="Negative momentum")



plt.xlabel(r"Momentum coefficient: $\mu$", fontsize=14)
plt.ylabel(r"Overshoot factor: $\gamma$", fontsize=14)
plt.yscale("symlog")
# plt.xlim(0, 1)
plt.ylim(top=200)
# plt.title("Overshoot equivalence for SGD")
plt.legend(loc="upper left")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

