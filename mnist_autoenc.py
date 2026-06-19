from torch import nn

class MNISTEncoder(nn.Module):
    def __init__(self, dropout_rate = 0.2, latent_dim = 64):
        super(MNISTEncoder, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 4, latent_dim)
        )
    
    def forward(self, x):
        x = self.convnet(x)
        x = self.linear(x)
        return x

class MNISTDecoder(nn.Module):
    def __init__(self, dropout_rate = 0.2, latent_dim = 64):
        super(MNISTDecoder, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 4, 16 * 7 * 7),
            nn.Unflatten(dim=1, unflattened_size=(16, 7, 7))
        )

        self.deconvnet = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.deconvnet(x)
        return x

class MNISTAutoencoder(nn.Module):
    def __init__(self, dropout_rate = 0.2, latent_dim = 64):
        super(MNISTAutoencoder, self).__init__()
        self.encoder = MNISTEncoder(dropout_rate, latent_dim)
        self.decoder = MNISTDecoder(dropout_rate, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed