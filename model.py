import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [3,64,64] -> [32,32,32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [32,32,32] -> [64,16,16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [64,16,16] -> [128,8,8]
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class Encoder_128(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [3,128,128] -> [32,64,64]
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [32,64,64] -> [64,32,32]
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [64,32,32] -> [128,16,16]
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class Decoder_128(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 16 * 16)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [128,16,16] -> [64,32,32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # [64,32,32] -> [32,64,64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # [32,64,64] -> [3,128,128]
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 16, 16)
        x = self.deconv(x)
        return x
    
class Dynamics(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, latent, action):
        action_onehot = F.one_hot(action, num_classes=4).float()
        x = torch.cat([latent, action_onehot], dim=1)
        return self.network(x)
    