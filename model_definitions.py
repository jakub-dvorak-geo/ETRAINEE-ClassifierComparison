"""Contains model definitions for deep semantic segmentation models."""
import torch
import torch.nn as nn


class SpectralNet(nn.Module):
    """1D CNN for classifying pixels based on the spectral response."""

    def __init__(self, n_channels, size, n_class, cuda):
        """
        Initialize the SpectralNet model.

        n_channels, int, number of input channel
        size, int list, size of the feature maps of convs for the encoder
        n_class = int,  the number of classes
        """
        # necessary for all classes extending the module class
        super(SpectralNet, self).__init__()

        self.maxpool = nn.MaxPool1d(2, return_indices=False)
        self.dropout = nn.Dropout(p=0.5, inplace=True)

        # Encoder layer definitions
        def c_1d(inch, outch, k_size=3, pad=1, pad_mode='reflect', bias=False):
            """Create default conv layer."""
            return nn.Conv1d(inch, outch, kernel_size=k_size, padding=pad,
                             padding_mode=pad_mode, bias=bias)

        self.c1 = nn.Sequential(c_1d(n_channels, size[0]),
                                nn.BatchNorm1d(size[0]), nn.ReLU())
        self.c2 = nn.Sequential(c_1d(size[0], size[1]),
                                nn.BatchNorm1d(size[1]), nn.ReLU())
        self.c3 = nn.Sequential(c_1d(size[1], size[2]),
                                nn.BatchNorm1d(size[2]), nn.ReLU())
        self.c4 = nn.Sequential(c_1d(size[2], size[3]),
                                nn.BatchNorm1d(size[3]), nn.ReLU())
        self.c5 = nn.Sequential(c_1d(size[3], size[4]),
                                nn.BatchNorm1d(size[4]), nn.ReLU())
        self.c6 = nn.Sequential(c_1d(size[4], size[5]),
                                nn.BatchNorm1d(size[5]), nn.ReLU())
        self.c7 = nn.Sequential(c_1d(size[5], size[6]),
                                nn.BatchNorm1d(size[6]), nn.ReLU())
        self.c8 = nn.Sequential(c_1d(size[6], size[7]),
                                nn.BatchNorm1d(size[7]), nn.ReLU())
        self.c9 = nn.Sequential(c_1d(size[7], size[8]),
                                nn.BatchNorm1d(size[8]), nn.ReLU())
        self.c10 = nn.Sequential(c_1d(size[8], size[9]),
                                 nn.BatchNorm1d(size[9]), nn.ReLU())
        # Final classifying layer
        self.classifier = c_1d(size[9], n_class, 1, padding=0)

        # Weight initialization
        self.c1[0].apply(self.init_weights)
        self.c2[0].apply(self.init_weights)
        self.c3[0].apply(self.init_weights)
        self.c4[0].apply(self.init_weights)
        self.c5[0].apply(self.init_weights)
        self.c6[0].apply(self.init_weights)
        self.c7[0].apply(self.init_weights)
        self.c8[0].apply(self.init_weights)
        self.c9[0].apply(self.init_weights)
        self.c10[0].apply(self.init_weights)

        self.classifier.apply(self.init_weights)

        if cuda:    # Put the model on GPU memory
            self.cuda()

    def init_weights(self, layer):  # gaussian init for the conv layers
        """Initialise layer weights."""
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Define model structure."""
        # Encoder
        # level 1
        x1 = self.c2(self.c1(input_data))
        x2 = self.maxpool(x1)
        # level 2
        x3 = self.c4(self.c3(x2))
        x4 = self.maxpool(x3)
        # level 3
        x5 = self.c6(self.c5(x4))
        x6 = self.maxpool(x5)
        # Level 4
        x7 = self.c8(self.c7(x6))
        x8 = self.maxpool(x7)
        # Level 5
        x9 = self.c10(self.c9(x8))

        # Output
        out = self.classifier(self.dropout(x9))
        return out


class UNet(nn.Module):
    """U-Net for semantic segmentation."""

    def __init__(self, n_channels, size_e, size_d, n_class, cuda):
        """
        Initialize the U-Net model.

        n_channels, int, number of input channel
        size_e, int list, size of the feature maps of convs for the encoder
        size_d, int list, size of the feature maps of convs for the decoder
        n_class = int,  the number of classes
        """
        super(UNet, self).__init__(
        )  # necessary for all classes extending the module class

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        def c_2d(inch, outch, k_size=3, pad=1, pad_mode='reflect', bias=False):
            """Create default conv layer."""
            return nn.Conv2d(inch, outch, kernel_size=k_size, padding=pad,
                             padding_mode=pad_mode, bias=bias)

        # Encoder layer definitions
        self.c1 = nn.Sequential(c_2d(n_channels, size_e[0]),
                                nn.BatchNorm2d(size_e[0]), nn.ReLU())
        self.c2 = nn.Sequential(c_2d(size_e[0], size_e[1]),
                                nn.BatchNorm2d(size_e[1]), nn.ReLU())
        self.c3 = nn.Sequential(c_2d(size_e[1], size_e[2]),
                                nn.BatchNorm2d(size_e[2]), nn.ReLU())
        self.c4 = nn.Sequential(c_2d(size_e[2], size_e[3]),
                                nn.BatchNorm2d(size_e[3]), nn.ReLU())
        self.c5 = nn.Sequential(c_2d(size_e[3], size_e[4]),
                                nn.BatchNorm2d(size_e[4]), nn.ReLU())
        self.c6 = nn.Sequential(c_2d(size_e[4], size_e[5]),
                                nn.BatchNorm2d(size_e[5]), nn.ReLU())
        self.c7 = nn.Sequential(c_2d(size_e[5], size_e[6]),
                                nn.BatchNorm2d(size_e[6]), nn.ReLU())
        self.c8 = nn.Sequential(c_2d(size_e[6], size_e[7]),
                                nn.BatchNorm2d(size_e[7]), nn.ReLU())
        self.c9 = nn.Sequential(c_2d(size_e[7], size_e[8]),
                                nn.BatchNorm2d(size_e[8]), nn.ReLU())
        self.c10 = nn.Sequential(c_2d(size_e[8], size_e[9]),
                                 nn.BatchNorm2d(size_e[9]), nn.ReLU())
        # Decoder layer definitions
        self.c11 = nn.ConvTranspose2d(size_e[9], int(size_d[0]/2),
                                      kernel_size=2, stride=2)
        self.c12 = nn.Sequential(c_2d(size_d[0], size_d[1]),
                                 nn.BatchNorm2d(size_d[1]), nn.ReLU())
        self.c13 = nn.Sequential(c_2d(size_d[1], size_d[2]),
                                 nn.BatchNorm2d(size_d[2]), nn.ReLU())
        self.c14 = nn.ConvTranspose2d(size_d[2], int(size_d[3]/2),
                                      kernel_size=2, stride=2)
        self.c15 = nn.Sequential(c_2d(size_d[3], size_d[4]),
                                 nn.BatchNorm2d(size_d[4]), nn.ReLU())
        self.c16 = nn.Sequential(c_2d(size_d[4], size_d[5]),
                                 nn.BatchNorm2d(size_d[5]), nn.ReLU())
        self.c17 = nn.ConvTranspose2d(size_d[5], int(size_d[6]/2),
                                      kernel_size=2, stride=2)
        self.c18 = nn.Sequential(c_2d(size_d[6], size_d[7]),
                                 nn.BatchNorm2d(size_d[7]), nn.ReLU())
        self.c19 = nn.Sequential(c_2d(size_d[7], size_d[8]),
                                 nn.BatchNorm2d(size_d[8]), nn.ReLU())
        self.c20 = nn.ConvTranspose2d(size_d[8], int(size_d[9]/2),
                                      kernel_size=2, stride=2)
        self.c21 = nn.Sequential(c_2d(size_d[9], size_d[10]),
                                 nn.BatchNorm2d(size_d[10]), nn.ReLU())
        self.c22 = nn.Sequential(c_2d(size_d[10], size_d[11]),
                                 nn.BatchNorm2d(size_d[11]), nn.ReLU())

        # Final classifying layer
        self.classifier = nn.Conv2d(size_d[11], n_class, 1, padding=0)

        # Weight initialization
        self.c1[0].apply(self.init_weights)
        self.c2[0].apply(self.init_weights)
        self.c3[0].apply(self.init_weights)
        self.c4[0].apply(self.init_weights)
        self.c5[0].apply(self.init_weights)
        self.c6[0].apply(self.init_weights)
        self.c7[0].apply(self.init_weights)
        self.c8[0].apply(self.init_weights)
        self.c9[0].apply(self.init_weights)
        self.c10[0].apply(self.init_weights)

        self.c12[0].apply(self.init_weights)
        self.c13[0].apply(self.init_weights)

        self.c15[0].apply(self.init_weights)
        self.c16[0].apply(self.init_weights)

        self.c18[0].apply(self.init_weights)
        self.c19[0].apply(self.init_weights)

        self.c21[0].apply(self.init_weights)
        self.c22[0].apply(self.init_weights)
        self.classifier.apply(self.init_weights)

        if cuda:    # Put the model on GPU memory
            self.cuda()

    def init_weights(self, layer):  # gaussian init for the conv layers
        """Initialise layer weights."""
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Define model structure."""
        # Encoder
        # level 1
        x1 = self.c2(self.c1(input_data))
        x2 = self.maxpool(x1)
        # level 2
        x3 = self.c4(self.c3(x2))
        x4 = self.maxpool(x3)
        # level 3
        x5 = self.c6(self.c5(x4))
        x6 = self.maxpool(x5)
        # Level 4
        x7 = self.c8(self.c7(x6))
        x8 = self.maxpool(x7)
        # Level 5
        x9 = self.c10(self.c9(x8))
        # Decoder
        # Level 4
        y8 = torch.cat((self.c11(x9), x7), 1)
        y7 = self.c13(self.c12(y8))
        # Level 3
        y6 = torch.cat((self.c14(y7), x5), 1)
        y5 = self.c16(self.c15(y6))
        # level 2
        y4 = torch.cat((self.c17(y5), x3), 1)
        y3 = self.c19(self.c18(y4))
        # level 1
        y2 = torch.cat((self.c20(y3), x1), 1)
        y1 = self.c22(self.c21(y2))
        # Output
        out = self.classifier(self.dropout(y1))
        return out


class SpectroSpatialNet(nn.Module):
    """3D Spectral-Spatial CNN for semantic segmentation."""

    def __init__(self, n_channels, size_e, size_d, n_class, cuda):
        """
        Initialize the SpectroSpatial model.

        n_channels, int, number of input channel
        size_e, int list, size of the feature maps of convs for the encoder
        size_d, int list, size of the feature maps of convs for the decoder
        n_class = int,  the number of classes
        """
        # necessary for all classes extending the module class
        super(SpectroSpatialNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        # Encoder layer definitions
        def c_3d(inch, outch, k_size=3, pad=1, pad_mode='reflect', bias=False):
            """Create default conv layer."""
            return nn.Conv3d(inch, outch, kernel_size=k_size, padding=pad,
                             padding_mode=pad_mode, bias=bias)

        self.c1 = nn.Sequential(c_3d(n_channels, size_e[0]),
                                nn.BatchNorm2d(size_e[0]), nn.ReLU())
        self.c2 = nn.Sequential(c_3d(size_e[0], size_e[1]),
                                nn.BatchNorm2d(size_e[1]), nn.ReLU())
        self.c3 = nn.Sequential(c_3d(size_e[1], size_e[2]),
                                nn.BatchNorm2d(size_e[2]), nn.ReLU())
        self.c4 = nn.Sequential(c_3d(size_e[2], size_e[3]),
                                nn.BatchNorm2d(size_e[3]), nn.ReLU())
        self.c5 = nn.Sequential(c_3d(size_e[3], size_e[4]),
                                nn.BatchNorm2d(size_e[4]), nn.ReLU())
        self.c6 = nn.Sequential(c_3d(size_e[4], size_e[5]),
                                nn.BatchNorm2d(size_e[5]), nn.ReLU())
        self.c7 = nn.Sequential(c_3d(size_e[5], size_e[6]),
                                nn.BatchNorm2d(size_e[6]), nn.ReLU())
        self.c8 = nn.Sequential(c_3d(size_e[6], size_e[7]),
                                nn.BatchNorm2d(size_e[7]), nn.ReLU())
        self.c9 = nn.Sequential(c_3d(size_e[7], size_e[8]),
                                nn.BatchNorm2d(size_e[8]), nn.ReLU())
        self.c10 = nn.Sequential(c_3d(size_e[8], size_e[9]),
                                 nn.BatchNorm2d(size_e[9]), nn.ReLU())
        # Decoder layer definitions
        self.c11 = nn.ConvTranspose2d(size_e[9], int(size_d[0]/2),
                                      kernel_size=2, stride=2)
        self.c12 = nn.Sequential(c_3d(size_d[0], size_d[1]),
                                 nn.BatchNorm2d(size_d[1]), nn.ReLU())
        self.c13 = nn.Sequential(c_3d(size_d[1], size_d[2]),
                                 nn.BatchNorm2d(size_d[2]), nn.ReLU())
        self.c14 = nn.ConvTranspose2d(size_d[2], int(size_d[3]/2),
                                      kernel_size=2, stride=2)
        self.c15 = nn.Sequential(c_3d(size_d[3], size_d[4]),
                                 nn.BatchNorm2d(size_d[4]), nn.ReLU())
        self.c16 = nn.Sequential(c_3d(size_d[4], size_d[5]),
                                 nn.BatchNorm2d(size_d[5]), nn.ReLU())
        self.c17 = nn.ConvTranspose2d(size_d[5], int(size_d[6]/2),
                                      kernel_size=2, stride=2)
        self.c18 = nn.Sequential(c_3d(size_d[6], size_d[7]),
                                 nn.BatchNorm2d(size_d[7]), nn.ReLU())
        self.c19 = nn.Sequential(c_3d(size_d[7], size_d[8]),
                                 nn.BatchNorm2d(size_d[8]), nn.ReLU())
        self.c20 = nn.ConvTranspose2d(size_d[8], int(size_d[9]/2),
                                      kernel_size=2, stride=2)
        self.c21 = nn.Sequential(c_3d(size_d[9], size_d[10]),
                                 nn.BatchNorm2d(size_d[10]), nn.ReLU())
        self.c22 = nn.Sequential(c_3d(size_d[10], size_d[11]),
                                 nn.BatchNorm2d(size_d[11]), nn.ReLU())

        # Final classifying layer
        self.classifier = c_3d(
            size_d[11], n_class, 1, padding=0)

        # Weight initialization
        self.c1[0].apply(self.init_weights)
        self.c2[0].apply(self.init_weights)
        self.c3[0].apply(self.init_weights)
        self.c4[0].apply(self.init_weights)
        self.c5[0].apply(self.init_weights)
        self.c6[0].apply(self.init_weights)
        self.c7[0].apply(self.init_weights)
        self.c8[0].apply(self.init_weights)
        self.c9[0].apply(self.init_weights)
        self.c10[0].apply(self.init_weights)

        self.c12[0].apply(self.init_weights)
        self.c13[0].apply(self.init_weights)

        self.c15[0].apply(self.init_weights)
        self.c16[0].apply(self.init_weights)

        self.c18[0].apply(self.init_weights)
        self.c19[0].apply(self.init_weights)

        self.c21[0].apply(self.init_weights)
        self.c22[0].apply(self.init_weights)
        self.classifier.apply(self.init_weights)

        if cuda:    # Put the model on GPU memory
            self.cuda()

    def init_weights(self, layer):  # gaussian init for the conv layers
        """Initialise layer weights."""
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Define model structure."""
        # Encoder
        # level 1
        x1 = self.c2(self.c1(input_data))
        x2 = self.maxpool(x1)
        # level 2
        x3 = self.c4(self.c3(x2))
        x4 = self.maxpool(x3)
        # level 3
        x5 = self.c6(self.c5(x4))
        x6 = self.maxpool(x5)
        # Level 4
        x7 = self.c8(self.c7(x6))
        x8 = self.maxpool(x7)
        # Level 5
        x9 = self.c10(self.c9(x8))
        # Decoder
        # Level 4
        y8 = torch.cat((self.c11(x9), x7), 1)
        y7 = self.c13(self.c12(y8))
        # Level 3
        y6 = torch.cat((self.c14(y7), x5), 1)
        y5 = self.c16(self.c15(y6))
        # level 2
        y4 = torch.cat((self.c17(y5), x3), 1)
        y3 = self.c19(self.c18(y4))
        # level 1
        y2 = torch.cat((self.c20(y3), x1), 1)
        y1 = self.c22(self.c21(y2))
        # Output
        out = self.classifier(self.dropout(y1))
        return out
