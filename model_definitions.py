"""Contains model definitions for deep semantic segmentation models."""
import torch
import torch.nn as nn


class SpectralNet(nn.Module):
    """1D CNN for classifying pixels based on the spectral response."""

    def __init__(self, args):
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

        self.n_channels = args['n_channel']
        self.size = args['layer_width']
        self.n_class = args['n_class']
        self.is_cuda = args['cuda']

        # Encoder layer definitions
        def conv_layer_1d(in_ch, out_ch, k_size=3, conv_bias=True):
            """Create default conv layer."""
            return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=k_size,
                                           bias=conv_bias),
                                 nn.BatchNorm1d(out_ch), nn.ReLU())

        def fconnected_layer(in_ch, out_ch, mlp_bias=True):
            """Create default linear layer."""
            return nn.Sequential(nn.Linear(in_ch, out_ch, bias=mlp_bias),
                                 nn.BatchNorm1d(out_ch), nn.ReLU())

        self.c1 = conv_layer_1d(self.n_channels, self.size[0])
        self.c2 = conv_layer_1d(self.size[0], self.size[1])
        self.c3 = conv_layer_1d(self.size[1], self.size[2])

        self.c4 = conv_layer_1d(self.size[2], self.size[3])
        self.c5 = conv_layer_1d(self.size[3], self.size[4])
        self.c6 = conv_layer_1d(self.size[4], self.size[5])

        self.flatten = nn.Flatten()

        self.l1 = fconnected_layer(self.size[6], self.size[7])
        self.l2 = fconnected_layer(self.size[7], self.size[8])
        self.l3 = fconnected_layer(self.size[8], self.size[9])
        # Final classifying layer
        self.classifier = nn.Linear(self.size[9], self.n_class)

        # Weight initialization
        self.c1[0].apply(self.init_weights)
        self.c2[0].apply(self.init_weights)
        self.c3[0].apply(self.init_weights)
        self.c4[0].apply(self.init_weights)
        self.c5[0].apply(self.init_weights)
        self.c6[0].apply(self.init_weights)
        self.l1[0].apply(self.init_weights)
        self.l2[0].apply(self.init_weights)
        self.l3[0].apply(self.init_weights)

        self.classifier.apply(self.init_weights)

        if self.is_cuda:    # Put the model on GPU memory
            self.cuda()

    def init_weights(self, layer):  # gaussian init for the conv layers
        """Initialise layer weights."""
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Define model structure."""
        # Encoder
        # level 1
        x1 = self.c1(input_data)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.maxpool(x3)
        # level 2
        x5 = self.c4(x4)
        x6 = self.c5(x5)
        x7 = self.c6(x6)
        x8 = self.maxpool(x7)
        # mlp
        x9 = self.flatten(x8)
        x10 = self.l1(x9)
        x11 = self.l2(x10)
        x12 = self.l3(x11)
        # Output
        out = self.classifier(self.dropout(x12))

        return out


class UNet(nn.Module):
    """U-Net for semantic segmentation."""

    def __init__(self, args):
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

        self.n_channels = args['n_channel']
        self.size_e = args['size_e']
        self.size_d = args['size_d']
        self.n_class = args['n_class']
        self.is_cuda = args['cuda']

        def c_2d(inch, outch, k_size=3, pad=1, pad_mode='reflect', bias=False):
            """Create default conv layer."""
            return nn.Conv2d(inch, outch, kernel_size=k_size, padding=pad,
                             padding_mode=pad_mode, bias=bias)

        # Encoder layer definitions
        self.c1 = nn.Sequential(c_2d(self.n_channels, self.size_e[0]),
                                nn.BatchNorm2d(self.size_e[0]), nn.ReLU())
        self.c2 = nn.Sequential(c_2d(self.size_e[0], self.size_e[1]),
                                nn.BatchNorm2d(self.size_e[1]), nn.ReLU())
        self.c3 = nn.Sequential(c_2d(self.size_e[1], self.size_e[2]),
                                nn.BatchNorm2d(self.size_e[2]), nn.ReLU())
        self.c4 = nn.Sequential(c_2d(self.size_e[2], self.size_e[3]),
                                nn.BatchNorm2d(self.size_e[3]), nn.ReLU())
        self.c5 = nn.Sequential(c_2d(self.size_e[3], self.size_e[4]),
                                nn.BatchNorm2d(self.size_e[4]), nn.ReLU())
        self.c6 = nn.Sequential(c_2d(self.size_e[4], self.size_e[5]),
                                nn.BatchNorm2d(self.size_e[5]), nn.ReLU())
        self.c7 = nn.Sequential(c_2d(self.size_e[5], self.size_e[6]),
                                nn.BatchNorm2d(self.size_e[6]), nn.ReLU())
        self.c8 = nn.Sequential(c_2d(self.size_e[6], self.size_e[7]),
                                nn.BatchNorm2d(self.size_e[7]), nn.ReLU())
        self.c9 = nn.Sequential(c_2d(self.size_e[7], self.size_e[8]),
                                nn.BatchNorm2d(self.size_e[8]), nn.ReLU())
        self.c10 = nn.Sequential(c_2d(self.size_e[8], self.size_e[9]),
                                 nn.BatchNorm2d(self.size_e[9]), nn.ReLU())
        # Decoder layer definitions
        self.c11 = nn.ConvTranspose2d(self.size_e[9], int(self.size_d[0]/2),
                                      kernel_size=2, stride=2)
        self.c12 = nn.Sequential(c_2d(self.size_d[0], self.size_d[1]),
                                 nn.BatchNorm2d(self.size_d[1]), nn.ReLU())
        self.c13 = nn.Sequential(c_2d(self.size_d[1], self.size_d[2]),
                                 nn.BatchNorm2d(self.size_d[2]), nn.ReLU())
        self.c14 = nn.ConvTranspose2d(self.size_d[2], int(self.size_d[3]/2),
                                      kernel_size=2, stride=2)
        self.c15 = nn.Sequential(c_2d(self.size_d[3], self.size_d[4]),
                                 nn.BatchNorm2d(self.size_d[4]), nn.ReLU())
        self.c16 = nn.Sequential(c_2d(self.size_d[4], self.size_d[5]),
                                 nn.BatchNorm2d(self.size_d[5]), nn.ReLU())
        self.c17 = nn.ConvTranspose2d(self.size_d[5], int(self.size_d[6]/2),
                                      kernel_size=2, stride=2)
        self.c18 = nn.Sequential(c_2d(self.size_d[6], self.size_d[7]),
                                 nn.BatchNorm2d(self.size_d[7]), nn.ReLU())
        self.c19 = nn.Sequential(c_2d(self.size_d[7], self.size_d[8]),
                                 nn.BatchNorm2d(self.size_d[8]), nn.ReLU())
        self.c20 = nn.ConvTranspose2d(self.size_d[8], int(self.size_d[9]/2),
                                      kernel_size=2, stride=2)
        self.c21 = nn.Sequential(c_2d(self.size_d[9], self.size_d[10]),
                                 nn.BatchNorm2d(self.size_d[10]), nn.ReLU())
        self.c22 = nn.Sequential(c_2d(self.size_d[10], self.size_d[11]),
                                 nn.BatchNorm2d(self.size_d[11]), nn.ReLU())

        # Final classifying layer
        self.classifier = nn.Conv2d(self.size_d[11], self.n_class,
                                    1, padding=0)

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

        if self.is_cuda:    # Put the model on GPU memory
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

    def __init__(self, args):
        """
        Initialize the SpectroSpatial model.

        n_channels, int, number of input channel
        size_e, int list, size of the feature maps of convs for the encoder
        size_d, int list, size of the feature maps of convs for the decoder
        n_class = int,  the number of classes
        """
        # necessary for all classes extending the module class
        super(SpectroSpatialNet, self).__init__()

        self.maxpool = nn.MaxPool3d(2, 2, return_indices=False)
        self.dropout = nn.Dropout3d(p=0.5, inplace=True)

        self.n_channels = args['n_channel']
        self.size_e = args['size_e']
        self.size_d = args['size_d']
        self.n_class = args['n_class']
        self.is_cuda = args['cuda']

        # Encoder layer definitions
        def c_en_3d(in_ch, out_ch, k_size=3, pad=1, pad_mode='zeros',
                    bias=False):
            """Create default conv layer for the encoder."""
            return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=k_size,
                                           padding=pad, padding_mode=pad_mode,
                                           bias=bias),
                                 nn.BatchNorm3d(out_ch), nn.ReLU())

        def c_de_2d(in_ch, out_ch, k_size=3, pad=1, pad_mode='zeros',
                    bias=False):
            """Create default conv layer for the decoder."""
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=k_size,
                                           padding=pad, padding_mode=pad_mode,
                                           bias=bias),
                                 nn.BatchNorm2d(out_ch), nn.ReLU())

        def c_3d(inch, outch, k_size=3, pad=1, pad_mode='zeros',
                 bias=False):
            """Create default conv layer."""
            return nn.Conv3d(inch, outch, kernel_size=k_size, padding=pad,
                             padding_mode=pad_mode, bias=bias)

        self.c1 = c_en_3d(self.n_channels, self.size_e[0])
        self.c2 = c_en_3d(self.size_e[0], self.size_e[1])
        self.c3 = c_en_3d(self.size_e[1], self.size_e[2])
        self.c4 = c_en_3d(self.size_e[2], self.size_e[3])
        self.c5 = c_en_3d(self.size_e[3], self.size_e[4])
        self.c6 = c_en_3d(self.size_e[4], self.size_e[5])
        self.c7 = c_en_3d(self.size_e[5], self.size_e[6])
        self.c8 = c_en_3d(self.size_e[6], self.size_e[7])
        self.c9 = c_en_3d(self.size_e[7], self.size_e[8])
        self.c10 = c_en_3d(self.size_e[8], self.size_e[9])

        # Decoder layer definitions
        self.c11 = nn.ConvTranspose2d(int(self.size_e[9]) * 3, self.size_d[0],
                                      kernel_size=2, stride=2)
        self.c12 = c_de_2d(self.size_d[0], self.size_d[1])
        self.c13 = c_de_2d(self.size_d[1], self.size_d[2])
        self.c14 = nn.ConvTranspose2d(self.size_d[2], self.size_d[3],
                                      kernel_size=2, stride=2)
        self.c15 = c_de_2d(self.size_d[3], self.size_d[4])
        self.c16 = c_de_2d(self.size_d[4], self.size_d[5])
        self.c17 = nn.ConvTranspose2d(self.size_d[5], self.size_d[6],
                                      kernel_size=2, stride=2)
        self.c18 = c_de_2d(self.size_d[6], self.size_d[7])
        self.c19 = c_de_2d(self.size_d[7], self.size_d[8])
        self.c20 = nn.ConvTranspose2d(self.size_d[8], self.size_d[9],
                                      kernel_size=2, stride=2)
        self.c21 = c_de_2d(self.size_d[9], self.size_d[10])
        self.c22 = c_de_2d(self.size_d[10], self.size_d[11])

        # Final classifying layer
        self.classifier = nn.Conv2d(self.size_d[11], self.n_class,
                                    1, padding=0)

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

        if self.is_cuda:    # Put the model on GPU memory
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
        y9 = torch.flatten(x9, start_dim=1, end_dim=2)
        y8 = self.c11(y9)
        y7 = self.c13(self.c12(y8))
        # Level 3
        y6 = self.c14(y7)
        y5 = self.c16(self.c15(y6))
        # level 2
        y4 = self.c17(y5)
        y3 = self.c19(self.c18(y4))
        # level 1
        y2 = self.c20(y3)
        y1 = self.c22(self.c21(y2))
        # Output
        out = self.classifier(self.dropout(y1))
        return out


class KrakonosNet(nn.Module):
    """KrakonosNet network based on U-Net for semantic segmentation."""

    def __init__(self, n_channels, encoder_conv_width, decoder_conv_width,
                 n_class, cuda):
        """
        Initialize the network, define structure.

        n_channels, int, number of input channel
        encoder_conv_width, int list, size of encoder conv feature maps
        decoder_conv_width, int list, size of decoder conv feature maps
        n_class = int,  the number of classes
        """
        super(KrakonosNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        # encoder
        self.c1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[0]), nn.PReLU())
        self.c2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0], encoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[1]), nn.PReLU())
        self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1], encoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[2]), nn.PReLU())
        self.c4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2], encoder_conv_width[3], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[3]), nn.PReLU())
        self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3], encoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[4]), nn.PReLU())
        self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4], encoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[5]), nn.PReLU())
        self.c7 = nn.Sequential(nn.Conv2d(encoder_conv_width[5], encoder_conv_width[6], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[6]), nn.PReLU())
        self.c8 = nn.Sequential(nn.Conv2d(encoder_conv_width[6], encoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[7]), nn.PReLU())
        self.c9 = nn.Sequential(nn.Conv2d(encoder_conv_width[7], encoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[8]), nn.PReLU())
        self.c10 = nn.Sequential(nn.Conv2d(encoder_conv_width[8], encoder_conv_width[9], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            encoder_conv_width[9]), nn.PReLU())
        # decoder
        self.c11 = nn.ConvTranspose2d(encoder_conv_width[9], int(
            decoder_conv_width[0]/2), kernel_size=2, stride=2)
        self.c12 = nn.Sequential(nn.Conv2d(decoder_conv_width[0], decoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[1]), nn.PReLU())
        self.c13 = nn.Sequential(nn.Conv2d(decoder_conv_width[1], decoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[2]), nn.PReLU())
        self.c14 = nn.ConvTranspose2d(decoder_conv_width[2], int(
            decoder_conv_width[3]/2), kernel_size=2, stride=2)
        self.c15 = nn.Sequential(nn.Conv2d(decoder_conv_width[3], decoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[4]), nn.PReLU())
        self.c16 = nn.Sequential(nn.Conv2d(decoder_conv_width[4], decoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[5]), nn.PReLU())
        self.c17 = nn.ConvTranspose2d(decoder_conv_width[5], int(
            decoder_conv_width[6]/2), kernel_size=2, stride=2)
        self.c18 = nn.Sequential(nn.Conv2d(decoder_conv_width[6], decoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[7]), nn.PReLU())
        self.c19 = nn.Sequential(nn.Conv2d(decoder_conv_width[7], decoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[8]), nn.PReLU())
        self.c20 = nn.ConvTranspose2d(decoder_conv_width[8], int(
            decoder_conv_width[9]/2), kernel_size=2, stride=2)
        self.c21 = nn.Sequential(nn.Conv2d(decoder_conv_width[9], decoder_conv_width[10], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(
            decoder_conv_width[10]), nn.PReLU())
        self.c22 = nn.Sequential(nn.Conv2d(decoder_conv_width[10], decoder_conv_width[11], 3,
                                 padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[11]), nn.PReLU())

        # final classifying layer
        self.classifier = nn.Conv2d(
            decoder_conv_width[11], n_class, 1, padding=0)

        # weight initialization
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

        if cuda:  # put the model on the GPU memory
            self.cuda()

    def init_weights(self, layer):
        """Gaussian init for the conv layers."""
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Runnning inference."""
        # encoder
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
        # decoder
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
        # output
        out = self.classifier(self.dropout(y1))

        return out
