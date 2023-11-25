"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm_activation=True, **kwargs):
        """
        CNN block consisting of convolution, batch normalization, and leaky ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            batchnorm_activation (bool): Flag to use batch normalization and leaky ReLU activation.
            **kwargs: Additional arguments for the convolutional layer.
        """
        super(CNNBlock, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batchnorm_activation, **kwargs)
        
        # Batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Flag to control whether to use batchnorm activation
        self.batchnorm_activation = batchnorm_activation
        
    def forward(self, x):
        if self.batchnorm_activation:
            return self.leaky_relu(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)
              
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        """
        Residual block consisting of multiple repeated CNN blocks.

        Args:
            channels (int): Number of input channels.
            use_residual (bool): Flag to use residual connections.
            num_repeats (int): Number of times to repeat the internal CNN blocks.
        """
        super(ResidualBlock, self).__init__()
        
        # List to store repeated CNN blocks
        self.layers = nn.ModuleList()
        
        # Create and add repeated CNN blocks to the list
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels //2, channels, kernel_size=3, padding=1)
                )
            ]
            
        # Flag to control whether to use residual connections
        self.use_residual = use_residual
        
        self.num_repeats = num_repeats
        
        
    def forward(self, x):
        for layer in self.layers:
            # Apply repeated CNN blocks
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
                
        return x 
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Scale Prediction module responsible for generating predictions.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for prediction.
        """
        super(ScalePrediction, self).__init__()

        # Sequential block for prediction
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, batchnorm_activation=False, kernel_size=1),
        )

        # Number of classes for prediction
        self.num_classes = num_classes

    def forward(self, x):
    
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) # (batch, prediction, height, width, channels)
        )

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        """
        YOLOv3 architecture for object detection.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for prediction.
        """
        
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        
    def forward(self, x):
        outputs = []  # List to store predictions for each scale
        route_connections = []  # List to store feature maps for skip connections
        
        # Iterate through each layer in the network
        for layer in self.layers:
            # If the current layer is a ScalePrediction, append its output to the list and continue
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            # Forward pass through the current layer
            x = layer(x)
            
            # If the layer is a ResidualBlock with 8 repeats, store the feature map for skip connection
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
                
            # If the layer is an Upsample layer, concatenate the current feature map with the stored skip connection
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
                
        return outputs
    
    def _create_conv_layers(self):
        """
        Create convolutional layers based on the YOLOv3 configuration.

        Returns:
            nn.ModuleList: List of convolutional layers.
        """
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
                
            if isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats
                    )
                )
                
            if isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                    
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3
                    
        return layers
    
    
if __name__ == "__main__":
    # Configuration
    num_classes = 20
    IMAGE_SIZE = 416

    # Create an instance of the YOLOv3 model
    model = YOLOv3(num_classes=num_classes)

    # Sample input tensor
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))

    # Ensure the output shapes match the expected dimensions for each scale
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)

    print("Success!")
