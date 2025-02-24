from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super(VAE, self).__init__()
        
        # Encoder: Convolutional layers to compress the input
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
        )
        
        # Fully connected layers to map to the latent space (mean and logvar for reparameterization trick)
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # Decoder: Fully connected layer to expand from latent_dim to the correct shape for the decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Decoder: Transposed Convolutional layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: (1, 28, 28)
            nn.Flatten(),
            nn.Linear(32 * 32, 28 * 28),
            nn.Sigmoid(),  # To ensure output is between 0 and 1
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 4, 4)  # Reshape to start decoding
        x = self.decoder(x)
        return x

    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     return self.decode(z), mu, logvar
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        
        labels = x.clone()
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        
        recon_loss = torch.sum((decoded.view(x.shape[0], -1) - labels.view(x.shape[0], -1)) ** 2, 1)
        
        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)
        # Total Loss
        loss = torch.mean(recon_loss) + kl_divergence.mean()

        return {'loss': loss, 'logits': decoded}



# # Loss function for VAE (reconstruction + KL divergence)
# def vae_loss(reconstructed, original, mu, logvar):
#     # Reconstruction loss
#     recon_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
#     # KL divergence
#     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl_div
