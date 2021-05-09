# Multi-Dilation Model by self-implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLDRnet(nn.Module):
    def __init__(self, freq_bin = 360):
        super(MLDRnet, self).__init__()

        # Encoder
        self.encoder_bn = nn.BatchNorm2d(3)
        self.encoder_c2_1 = nn.Conv2d(3, 3, 3, padding=1, stride=2)
        self.encoder_c3_1 = nn.Conv2d(3, 3, 3, padding=1, stride=2)

        self.encoder_c1_1 = nn.Conv2d(10, 10, 3, padding=1, stride=2)
        self.encoder_c1_2 = nn.Conv2d(10, 10, 3, padding=1, stride=2)

        self.encoder_c2_2 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)
        self.encoder_c2_3 = nn.Conv2d(10, 10, 3, padding=1, stride=2)

        self.encoder_c3_2 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)
        self.encoder_c3_3 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)

        self.encoder_c2_4 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)
        self.encoder_c3_4 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)
        self.encoder_c3_5 = nn.ConvTranspose2d(10, 10, 1, output_padding=1, stride=2)

        self.encoder_final = nn.Conv2d(30, 10, 1)

        # Decoder
        self.decoder_bn = nn.BatchNorm2d(10)
        self.decoder_c1 = nn.Sequential( 
            nn.Conv2d(10, 10, 3, padding=1),
            nn.SELU()
        )
        
        self.decoder_bm = nn.Sequential(
            nn.AvgPool2d((freq_bin, 1)),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 1, 3, padding=1),
            nn.SELU()    
        ) 
        
        self.decoder_final = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=1),
            nn.SELU(),
            nn.Conv2d(10, 1, 3, padding=1),
            nn.SELU()
        )
        # Multi-Dilation ModuleList
        self.md_bn_1 = nn.ModuleList([
            nn.BatchNorm2d(3),
            nn.BatchNorm2d(3),
            nn.BatchNorm2d(3),
            nn.BatchNorm2d(30),
            nn.BatchNorm2d(30),
            nn.BatchNorm2d(30)
        ])
        self.md_bn_2 = nn.ModuleList([
            nn.BatchNorm2d(13),
            nn.BatchNorm2d(13),
            nn.BatchNorm2d(13),
            nn.BatchNorm2d(40),
            nn.BatchNorm2d(40),
            nn.BatchNorm2d(40)
        ])
        self.md_bn_3 = nn.ModuleList([
            nn.BatchNorm2d(23),
            nn.BatchNorm2d(23),
            nn.BatchNorm2d(23),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50)
        ])
        self.md_c1 = nn.ModuleList([
            nn.Conv2d(3, 10, 3, padding=3, dilation=3),
            nn.Conv2d(3, 10, 3, padding=3, dilation=3),
            nn.Conv2d(3, 10, 3, padding=3, dilation=3),
            nn.Conv2d(30, 10, 3, padding=3, dilation=3),
            nn.Conv2d(30, 10, 3, padding=3, dilation=3),
            nn.Conv2d(30, 10, 3, padding=3, dilation=3)
        ])
        self.md_c2 = nn.ModuleList([
            nn.Conv2d(13, 10, 3, padding=6, dilation=6),
            nn.Conv2d(13, 10, 3, padding=6, dilation=6),
            nn.Conv2d(13, 10, 3, padding=6, dilation=6),
            nn.Conv2d(40, 10, 3, padding=6, dilation=6),
            nn.Conv2d(40, 10, 3, padding=6, dilation=6),
            nn.Conv2d(40, 10, 3, padding=6, dilation=6)
            
        ])
        self.md_c3 = nn.ModuleList([
            nn.Conv2d(23, 10, 3, padding=6, dilation=6),
            nn.Conv2d(23, 10, 3, padding=6, dilation=6),
            nn.Conv2d(23, 10, 3, padding=6, dilation=6),
            nn.Conv2d(50, 10, 3, padding=6, dilation=6),
            nn.Conv2d(50, 10, 3, padding=6, dilation=6),
            nn.Conv2d(50, 10, 3, padding=6, dilation=6)
        ])
        self.md_act1 = nn.SELU()
        self.md_act2 = nn.SELU()
        self.md_act3 = nn.SELU()

        self.softmax = nn.Softmax(dim=2)

    def encoder(self, x):
        x = self.encoder_bn(x)
        f1 = x
        f2 = self.encoder_c2_1(f1)
        f3 = self.encoder_c3_1(f2)
        # print("f1 f2 f3:", f1.shape, f2.shape, f3.shape)
        f1 = self.multi_dilation(f1, 0)
        f2 = self.multi_dilation(f2, 1)
        f3 = self.multi_dilation(f3, 2)
        # print("f1 f2 f3:", f1.shape, f2.shape, f3.shape)

        f1_2 = self.encoder_c1_1(f1)
        f1_3 = self.encoder_c1_2(f1_2)
        # print("f1_3", f1_3.shape)

        f2_1 = self.encoder_c2_2(f2)
        f2_3 = self.encoder_c2_3(f2)
        # print("f2_1 f2_3", f2_1.shape, f2_3.shape)

        f3_2 = self.encoder_c3_2(f3)
        f3_1 = self.encoder_c3_3(f3_2)
        # print("f3_2 f3_1", f3_2.shape, f3_1.shape)

        f1 = torch.cat([f1, f2_1, f3_1], dim = 1)
        f2 = torch.cat([f2, f1_2, f3_2], dim = 1)
        f3 = torch.cat([f3, f1_3, f2_3], dim = 1)
        # print("f1 f2 f3:", f1.shape, f2.shape, f3.shape)

        f1 = self.multi_dilation(f1, 3)
        f2 = self.multi_dilation(f2, 4)
        f3 = self.multi_dilation(f3, 5)
        # print("f1 f2 f3:", f1.shape, f2.shape, f3.shape)

        f2 = self.encoder_c2_4(f2)
        f3 = self.encoder_c3_4(f3)
        f3 = self.encoder_c3_5(f3)
        # print("f1 f2 f3:", f1.shape, f2.shape, f3.shape)
        final_x = torch.cat([f1, f2, f3], dim = 1)
        final_x = self.encoder_final(final_x)
        # print("final_x:", final_x.shape)
        return final_x

    def decoder(self, x):
        x = self.decoder_bn(x)
        x = self.decoder_c1(x)

        bm = self.decoder_bm(x)
        # print("bm:", bm.shape)

        final_x = self.decoder_final(x)
        final_x = torch.cat([bm, final_x], dim = -2)
        # print("final_x", final_x.shape)
        return final_x, bm


    def multi_dilation(self, x, i):
        x0 = x
        x1 = self.md_bn_1[i](x0)
        x1 = self.md_c1[i](x1)
        x1 = self.md_act1(x1)
        # print("x1:", x1.shape)

        x2 = torch.cat([x0, x1], dim = 1)
        x2 = self.md_bn_2[i](x2)
        x2 = self.md_c2[i](x2)
        x2 = self.md_act2(x2)
        # print("x2:", x2.shape)

        x3 = torch.cat([x0, x1, x2], dim = 1)
        x3 = self.md_bn_3[i](x3)
        x3 = self.md_c3[i](x3)
        x3 = self.md_act3(x3)
        # print("x3:", x3.shape)

        return x3

    def forward(self, x):
        x = self.encoder(x)
        output_pre, bm = self.decoder(x)
        output = self.softmax(output_pre)
        # print("output bm:", output.shape, bm.shape)
        # exit()
        return output, output_pre