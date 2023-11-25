import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolutionBlock, self).__init__()
        # Define a double convolution block with batch normalization and ReLU activation
        self.convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.convolution_block(x)
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the downsampling (encoder) blocks of the UNet
        for feature in features:
            self.downsampling_blocks.append(DoubleConvolutionBlock(in_channels, feature))
            in_channels = feature
            
        # Bottleneck layer
        self.bottleneck = DoubleConvolutionBlock(features[-1], features[-1]*2)
            
        # Define the upsampling (decoder) blocks of the UNet
        for feature in reversed(features):
            self.upsampling_blocks.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.upsampling_blocks.append(DoubleConvolutionBlock(feature*2, feature))
        
        # Final convolution layer
        self.final_convolution = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoding (Downsampling blocks)
        for block in self.downsampling_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pooling(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip_connections to add in decoding
        skip_connections = skip_connections[::-1]
        
        # Decoding (Upsampling blocks)
        for idx in range(0, len(self.upsampling_blocks), 2):
            x = self.upsampling_blocks[idx](x)
            
            skip_connection = skip_connections[idx//2]
            
            # Resize if needed to match the size of skip connection
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concatenated_skip = torch.cat((skip_connection, x), dim=1) # channel dimension
            x = self.upsampling_blocks[idx+1](concatenated_skip)
            
        return self.final_convolution(x)
    
    
def test_UNet():
    # Test UNet model
    x = torch.randn((8, 3, 161, 161))
    model = UNet(in_channels=3, out_channels=1)
    prediction = model(x)
    print(x.shape, prediction.shape)
    # assert prediction.shape == x.shape
    print(x)
    print(prediction)
    
    
if __name__ == "__main__":
    test_UNet()