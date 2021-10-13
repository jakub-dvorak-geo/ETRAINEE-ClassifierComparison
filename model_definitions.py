"""Contains model definitions for deep semantic segmentation models."""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net for semantic segmentation."""

    def __init__(self, n_channels, encoder_conv_width, decoder_conv_width, n_class, cuda):
        """
        Initialize the U-Net model.

        n_channels, int, number of input channel
        encoder_conv_width, int list, size of the feature maps of convs for the encoder
        decoder_conv_width, int list, size of the feature maps of convs for the decoder
        n_class = int,  the number of classes
        """
        super(UNet, self).__init__() #necessary for all classes extending the module class

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        # Encoder layer definitions
        self.c1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[0]), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0], encoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[1]), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1], encoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[2]), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2], encoder_conv_width[3], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[3]), nn.ReLU())
        self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3], encoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[4]), nn.ReLU())
        self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4], encoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[5]), nn.ReLU())
        self.c7 = nn.Sequential(nn.Conv2d(encoder_conv_width[5], encoder_conv_width[6], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[6]), nn.ReLU())
        self.c8 = nn.Sequential(nn.Conv2d(encoder_conv_width[6], encoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[7]), nn.ReLU())
        self.c9 = nn.Sequential(nn.Conv2d(encoder_conv_width[7], encoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[8]), nn.ReLU())
        self.c10 = nn.Sequential(nn.Conv2d(encoder_conv_width[8], encoder_conv_width[9], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[9]), nn.ReLU())
        # Decoder layer definitions
        self.c11 = nn.ConvTranspose2d(encoder_conv_width[9], int(decoder_conv_width[0]/2), kernel_size=2, stride=2)
        self.c12 = nn.Sequential(nn.Conv2d(decoder_conv_width[0], decoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[1]), nn.ReLU())
        self.c13 = nn.Sequential(nn.Conv2d(decoder_conv_width[1], decoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[2]), nn.ReLU())
        self.c14 = nn.ConvTranspose2d(decoder_conv_width[2], int(decoder_conv_width[3]/2), kernel_size=2, stride=2)
        self.c15 = nn.Sequential(nn.Conv2d(decoder_conv_width[3], decoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[4]), nn.ReLU())
        self.c16 = nn.Sequential(nn.Conv2d(decoder_conv_width[4], decoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[5]), nn.ReLU())
        self.c17 = nn.ConvTranspose2d(decoder_conv_width[5], int(decoder_conv_width[6]/2), kernel_size=2, stride=2)
        self.c18 = nn.Sequential(nn.Conv2d(decoder_conv_width[6], decoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[7]), nn.ReLU())
        self.c19 = nn.Sequential(nn.Conv2d(decoder_conv_width[7], decoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[8]), nn.ReLU())
        self.c20 = nn.ConvTranspose2d(decoder_conv_width[8], int(decoder_conv_width[9]/2), kernel_size=2, stride=2)
        self.c21 = nn.Sequential(nn.Conv2d(decoder_conv_width[9], decoder_conv_width[10], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[10]), nn.ReLU())
        self.c22 = nn.Sequential(nn.Conv2d(decoder_conv_width[10], decoder_conv_width[11], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[11]), nn.ReLU())

        # Final classifying layer
        self.classifier = nn.Conv2d(decoder_conv_width[11], n_class, 1, padding=0)

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
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """Function defines model structure and runs inferrence."""
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


class KrakonosNet(nn.Module):
    """KrakonosNet network based on U-Net for semantic segmentation."""

    def __init__(self, n_channels, encoder_conv_width, decoder_conv_width, n_class, cuda):
        """
        initialization function
        n_channels, int, number of input channel
        encoder_conv_width, int list, size of the feature maps of convs for the encoder
        decoder_conv_width, int list, size of the feature maps of convs for the decoder
        n_class = int,  the number of classes
        """
        super(KrakonosNet, self).__init__()  # necessary for all classes extending the module class

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        #encoder
        self.c1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[0]), nn.PReLU())
        self.c2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0], encoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[1]), nn.PReLU())
        self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1], encoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[2]), nn.PReLU())
        self.c4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2], encoder_conv_width[3], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[3]), nn.PReLU())
        self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3], encoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[4]), nn.PReLU())
        self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4], encoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[5]), nn.PReLU())
        self.c7 = nn.Sequential(nn.Conv2d(encoder_conv_width[5], encoder_conv_width[6], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[6]), nn.PReLU())
        self.c8 = nn.Sequential(nn.Conv2d(encoder_conv_width[6], encoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[7]), nn.PReLU())
        self.c9 = nn.Sequential(nn.Conv2d(encoder_conv_width[7], encoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[8]), nn.PReLU())
        self.c10 = nn.Sequential(nn.Conv2d(encoder_conv_width[8], encoder_conv_width[9], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(encoder_conv_width[9]), nn.PReLU())
        #decoder
        self.c11 = nn.ConvTranspose2d(encoder_conv_width[9], int(decoder_conv_width[0]/2), kernel_size=2, stride=2)
        self.c12 = nn.Sequential(nn.Conv2d(decoder_conv_width[0], decoder_conv_width[1], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[1]), nn.PReLU())
        self.c13 = nn.Sequential(nn.Conv2d(decoder_conv_width[1], decoder_conv_width[2], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[2]), nn.PReLU())
        self.c14 = nn.ConvTranspose2d(decoder_conv_width[2], int(decoder_conv_width[3]/2), kernel_size=2, stride=2)
        self.c15 = nn.Sequential(nn.Conv2d(decoder_conv_width[3], decoder_conv_width[4], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[4]), nn.PReLU())
        self.c16 = nn.Sequential(nn.Conv2d(decoder_conv_width[4], decoder_conv_width[5], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[5]), nn.PReLU())
        self.c17 = nn.ConvTranspose2d(decoder_conv_width[5], int(decoder_conv_width[6]/2), kernel_size=2, stride=2)
        self.c18 = nn.Sequential(nn.Conv2d(decoder_conv_width[6], decoder_conv_width[7], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[7]), nn.PReLU())
        self.c19 = nn.Sequential(nn.Conv2d(decoder_conv_width[7], decoder_conv_width[8], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[8]), nn.PReLU())
        self.c20 = nn.ConvTranspose2d(decoder_conv_width[8], int(decoder_conv_width[9]/2), kernel_size=2, stride=2)
        self.c21 = nn.Sequential(nn.Conv2d(decoder_conv_width[9], decoder_conv_width[10], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[10]), nn.PReLU())
        self.c22 = nn.Sequential(nn.Conv2d(decoder_conv_width[10], decoder_conv_width[11], 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(decoder_conv_width[11]), nn.PReLU())

        #final classifying layer
        self.classifier = nn.Conv2d(
            decoder_conv_width[11], n_class, 1, padding=0)

        #weight initialization

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

    def init_weights(self, layer):  # gaussian init for the conv layers
        nn.init.kaiming_normal_(
            layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_data):
        """
        the function called to run inference
        """
        #encoder
        #level 1
        x1 = self.c2(self.c1(input_data))
        x2 = self.maxpool(x1)
        #level 2
        x3 = self.c4(self.c3(x2))
        x4 = self.maxpool(x3)
        #level 3
        x5 = self.c6(self.c5(x4))
        x6 = self.maxpool(x5)
        #Level 4
        x7 = self.c8(self.c7(x6))
        x8 = self.maxpool(x7)
        #Level 5
        x9 = self.c10(self.c9(x8))
        #decoder
        #Level 4
        y8 = torch.cat((self.c11(x9), x7), 1)
        y7 = self.c13(self.c12(y8))
        #Level 3
        y6 = torch.cat((self.c14(y7), x5), 1)
        y5 = self.c16(self.c15(y6))
        #level 2
        y4 = torch.cat((self.c17(y5), x3), 1)
        y3 = self.c19(self.c18(y4))
        #level 1
        y2 = torch.cat((self.c20(y3), x1), 1)
        y1 = self.c22(self.c21(y2))
        #output
        out = self.classifier(self.dropout(y1))

        return out
