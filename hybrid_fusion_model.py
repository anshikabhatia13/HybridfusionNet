import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import VGGPerceptualLoss

class StackedAutoencoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[64, 128, 256]):
        super(StackedAutoencoder, self).__init__()
        
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),  # stride=2 for downsampling
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),  # stride=2 for downsampling
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),  # stride=2 for downsampling
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(True)
        )
        
        # Decoder layers
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Skip connection refinement
        self.refine1 = nn.Conv2d(hidden_dims[1] * 2, hidden_dims[1], kernel_size=1)
        self.refine2 = nn.Conv2d(hidden_dims[0] * 2, hidden_dims[0], kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections
        d1 = self.dec1(e3)
        # Refine and combine features
        d1 = self.refine1(torch.cat([d1, e2], dim=1))
        
        d2 = self.dec2(d1)
        # Refine and combine features
        d2 = self.refine2(torch.cat([d2, e1], dim=1))
        
        d3 = self.dec3(d2)
        
        return e3, d3

class TemporalLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, input_size)  # Add a linear layer to project back to input size
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.fc(lstm_out)  # Project back to original feature size
        return lstm_out

class FeatureFusion(nn.Module):
    def __init__(self, spatial_channels=256, temporal_channels=256):
        super(FeatureFusion, self).__init__()
        
        # Enhanced channel attention
        self.spatial_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(spatial_channels, spatial_channels // 16, 1),
            nn.ReLU(True),
            nn.Conv2d(spatial_channels // 16, spatial_channels, 1),
            nn.Sigmoid()
        )
        
        self.temporal_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(temporal_channels, temporal_channels // 16, 1),
            nn.ReLU(True),
            nn.Conv2d(temporal_channels // 16, temporal_channels, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale feature extraction
        self.spatial_conv1 = nn.Conv2d(spatial_channels, 128, 1)
        self.spatial_conv3 = nn.Conv2d(spatial_channels, 128, 3, padding=1)
        self.spatial_conv5 = nn.Conv2d(spatial_channels, 128, 5, padding=2)
        
        self.temporal_conv1 = nn.Conv2d(temporal_channels, 128, 1)
        self.temporal_conv3 = nn.Conv2d(temporal_channels, 128, 3, padding=1)
        self.temporal_conv5 = nn.Conv2d(temporal_channels, 128, 5, padding=2)
        
        # Progressive upsampling fusion layers
        self.fusion_stage1 = nn.Sequential(
            nn.Conv2d(768, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 384, 4, stride=2, padding=1),  # First upsampling
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        
        self.fusion_stage2 = nn.Sequential(
            nn.Conv2d(384, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 192, 4, stride=2, padding=1),  # Second upsampling
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        
        self.fusion_stage3 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 4, stride=2, padding=1),  # Third upsampling
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, spatial_features, temporal_features):
        # Apply channel attention
        spatial_att = self.spatial_ca(spatial_features)
        temporal_att = self.temporal_ca(temporal_features)
        
        spatial_features = spatial_features * spatial_att
        temporal_features = temporal_features * temporal_att
        
        # Multi-scale feature extraction
        s1 = self.spatial_conv1(spatial_features)
        s3 = self.spatial_conv3(spatial_features)
        s5 = self.spatial_conv5(spatial_features)
        
        t1 = self.temporal_conv1(temporal_features)
        t3 = self.temporal_conv3(temporal_features)
        t5 = self.temporal_conv5(temporal_features)
        
        # Concatenate all features
        fused = torch.cat([s1, s3, s5, t1, t3, t5], dim=1)
        
        # Progressive upsampling and fusion
        x = self.fusion_stage1(fused)
        x = self.fusion_stage2(x)
        x = self.fusion_stage3(x)
        
        # Final output
        output = self.final_conv(x)
        
        return output

class HybridFusionModel(nn.Module):
    def __init__(self, input_channels=3, sequence_length=3):
        super(HybridFusionModel, self).__init__()
        
        # Spatial feature extraction
        self.sae = StackedAutoencoder(input_channels=input_channels)
        
        # Temporal feature extraction
        self.lstm = TemporalLSTM()
        
        # Feature fusion
        self.fusion = FeatureFusion()
        
        # Enhanced color refinement
        self.color_enhancement = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.sequence_length = sequence_length
        
    def forward(self, x):
        batch_size = x.size(0)
        original_size = x.size()[3:]
        
        # Process each frame
        spatial_features = []
        temporal_features = []
        
        for i in range(self.sequence_length):
            frame = x[:, i]
            encoded, _ = self.sae(frame)
            spatial_features.append(encoded)
            
            # Prepare temporal features
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            temporal_input = avg_pool(encoded).view(batch_size, -1)
            temporal_features.append(temporal_input)
        
        # Process features
        spatial_features = torch.stack(spatial_features, dim=1)
        temporal_features = torch.stack(temporal_features, dim=1)
        
        temporal_out = self.lstm(temporal_features)
        temporal_out = temporal_out[:, -1]
        
        h, w = spatial_features.size(-2), spatial_features.size(-1)
        temporal_out = temporal_out.view(batch_size, -1, 1, 1).expand(-1, -1, h, w)
        
        # Fusion and enhancement
        fused = self.fusion(spatial_features[:, -1], temporal_out)
        enhanced = self.color_enhancement(fused)
        
        return enhanced

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        
    def forward(self, output, target):
        # Reconstruction loss
        recon_loss = self.mse_loss(output, target)
        
        # Structural similarity loss
        ssim_loss = self.ssim_loss(output, target)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(output, target)
        
        # Total loss
        total_loss = recon_loss + 0.1 * ssim_loss + 0.1 * perceptual_loss
        
        return total_loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = None  # Initialize as None, will be created on first forward pass
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self, window_size, channel, device):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)
        
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        # Create window on the same device as input if not already created
        if self.window is None or self.window.device != img1.device:
            self.window = self.create_window(self.window_size, channel, img1.device)
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.view(ssim_map.size(0), -1).mean(1) 