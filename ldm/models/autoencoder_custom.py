import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),  # Using ReLU for better stability
            nn.Conv2d(32, 72, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.encoder(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(72, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # Using ReLU for better stability
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  # Final convolution to refine output
        )
        self._initialize_weights()

    def forward(self, x):
        return self.decoder(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Autoencoder(nn.Module):
    def __init__(self, embed_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = torch.clamp(x, min=-1.,max=1.)
        return x

