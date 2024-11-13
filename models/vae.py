import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# According to: https://github.com/fsschneider/DeepOBS/blob/master/deepobs/tensorflow/testproblems/_vae.py
# TODO: Only working with input shape 28x28
class VAE(nn.Module):
    def __init__(self, latent_dim: int  = 8):
        super(VAE, self).__init__()
        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28 * 28, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 200),
        #     nn.ReLU()
        # )
        
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, padding='valid'),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(64, 64, 4, 2, padding='valid'),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(64, 64, 4, 1, padding='valid'),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Flatten(),
        )
        
        
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        
        # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 28 * 28),
        # )
        self.decoder_preprocess = nn.Sequential(
                nn.Linear(8, 24),
                nn.LeakyReLU(),
                nn.Linear(24, 49),
                nn.LeakyReLU(),
        )
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 64, 4, 2),
                nn.Dropout(0.2),
                nn.Conv2d(64, 64, 4, 1),
                nn.Dropout(0.2),
                nn.Conv2d(64, 64, 4, 1),
                nn.Flatten(),
                nn.Linear(6400, 28 * 28),
                nn.Sigmoid()
        )
        
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Labels is cloath type, which we do not use in generative model
    def forward(self, x, labels=None):
        labels = x.clone()
        
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)

        x_reconstructed = self.decoder(self.decoder_preprocess(z).view(-1, 1, 7, 7)).view(labels.shape)

        # recon_loss = nn.functional.binary_cross_entropy_with_logits(x_reconstructed, labels, reduction="sum")
        # recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, labels, reduction="sum")
        # recon_loss = torch.sum((x_reconstructed.view(-1) - labels.view(-1)) ** 2, 1)
        recon_loss = torch.sum((x_reconstructed.view(x.shape[0], -1) - labels.view(x.shape[0], -1)) ** 2, 1)
        
        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)
        # Total Loss
        loss = torch.mean(recon_loss) + kl_divergence.mean()

        return {'loss': loss, 'logits': x_reconstructed}