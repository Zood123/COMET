""" Full assembly of the parts to form the complete network """

from models.unet_parts import *
import torchvision.models as models
class UNetResNet18(nn.Module):
    def __init__(self, num_classes):
        super(UNetResNet18, self).__init__()
        # Load pre-trained ResNet18 model
        resnet18 = models.resnet18(pretrained=False)
        resnet18.fc = torch.nn.Linear(
                in_features=resnet18.fc.in_features, out_features=60)
        weights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"
        resnet18.load_state_dict(torch.load(weights_path))
        
        # Encoder (downsampling path)
        self.encoder = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        
        # Final convolution to get the output
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)
        
        # Decoder forward pass
        x = self.decoder(x)
        
        # Final convolution to get the output
        x = self.final_conv(x)
        
        return x



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # torch.Size([128, 3, 224, 224])
        x1 = self.inc(x)
        # torch.Size([128, 64, 224, 224])
        x2 = self.down1(x1)
        # torch.Size([128, 128, 112, 112])
        x3 = self.down2(x2)
        # torch.Size([128, 256, 56, 56])
        x4 = self.down3(x3)
        # torch.Size([128, 512, 28, 28])
        x5 = self.down4(x4)
        # torch.Size([128, 1024, 14, 14])

        x = self.up1(x5, x4)
        # torch.Size([128, 512, 28, 28])
        x = self.up2(x, x3)
        # torch.Size([128, 256, 56, 56])
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)