import torch
import torch.nn as nn
from resnext import ResNeXt101
import torch.nn.functional as F
resnext_101_32_path = 'resnext_101_32x4d.pth'


class convA(nn.Module):

    def __init__(self,inch,outch):
        super(convA, self).__init__()

        self.conv2 = nn.Conv2d(inch, outch, (3, 3), padding=1)
        self.batch2 = nn.BatchNorm2d(outch)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(inch, outch, (1, 1), padding=0)
        self.batch3 = nn.BatchNorm2d(outch)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(inch, outch, (5, 5), padding=2)
        self.batch4 = nn.BatchNorm2d(outch)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv7 = nn.Conv2d(inch, outch, (7, 7), padding=3)
        self.batch7 = nn.BatchNorm2d(outch)
        self.relu7 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(3 * outch, outch, 3, padding=1)
        self.conv6 = nn.BatchNorm2d(outch)
        self.relu6 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c2 = self.conv2(x)
        b2 = self.batch2(c2)
        r2 = self.relu2(b2)
        c3 = self.conv3(x)
        b3 = self.batch3(c3)
        r3 = self.relu3(b3)
        c4 = self.conv4(x)
        b4 = self.batch4(c4)
        r4 = self.relu4(b4)

        merge = torch.cat([r2, r3, r4], dim=1)
        c5 = self.conv5(merge)
        out1 = self.conv6(c5)
        out = self.relu6(out1)
        return out





class convZ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convZ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)

class SHADOW(nn.Module):
    def __init__(self):
        super(SHADOW, self).__init__()
        resnext = ResNeXt101()

        self.layer0 = resnext.layer0   #64  128*128
        self.layer1 = resnext.layer1   #256   64*64
        self.layer2 = resnext.layer2   #512   32*32
        self.layer3 = resnext.layer3   #1024   16*16



        self.conv1 = convA(3, 64) # 64 256*256
        self.pool1 = nn.MaxPool2d(2)   # 64 128*128    .........
        self.conv2 = convA(64, 128)#128 128*128
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = convA(128, 256)# 256 64*64    .........
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = convA(256, 512) #512 32*32      ........
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = convA(512, 1024) #1024 16*16    .........

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  #512 32*32
        self.conv6 = convZ(1024, 512)                          #512 32*32
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)   #256  64*64
        self.conv7 = convZ(512, 256)                           #256  64*64
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)   #128  128*128
        self.conv8 = convZ(256, 128)                           #128  128*128
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)    #64   256*256
        self.conv9 = convZ(128, 64)                            #64 256*256
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)          #32 256*256
        self.conv11 = nn.Conv2d(32, 1, 1)                      # 1 256*256

        self.convx1=nn.Conv2d(128, 64, 3, padding=1)          #64  128*128
        self.convx2 = nn.Conv2d(512, 256, 3, padding=1)       #256  64*64
        self.convx3 = nn.Conv2d(1024, 512, 3, padding=1)      #512   32*32
        self.convx4 = nn.Conv2d(2048, 1024, 3, padding=1)     # 1024  16*16

    def forward(self, x):
        layer0 = self.layer0(x)       #64    128
        layer1 = self.layer1(layer0)  #256   64
        layer2 = self.layer2(layer1)  #512   32
        layer3 = self.layer3(layer2)  #1024  16


        c1 = self.conv1(x)                 #////////////////// 64 256*256
        p1 = self.pool1(c1)# 64 128*128    .........
        t1 = torch.cat([p1,layer0], dim=1)  #//////////////////
        x1 = self.convx1(t1)                #64  128*128
        c2 = self.conv2(x1)                  #////////////////// #128 128*128
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)# 256 64*64    ......... #////////////////// 256 64*64
        t2 = torch.cat([c3, layer1], dim=1)  #//////////////////
        x2 = self.convx2(t2)                   ##256  64*64
        p3 = self.pool3(x2)
        c4 = self.conv4(p3)                      #512 32*32
        t3 = torch.cat([c4, layer2], dim=1)  #//////////////////
        x3 = self.convx3(t3)                   ##512   32*32
        p4 = self.pool4(x3)
        c5 = self.conv5(p4)                 #1024 16*16    .........
        t4 = torch.cat([c5, layer3], dim=1)   #//////////////////
        x4 = self.convx4(t4)                  # 1024  16*16



        up_6 = self.up6(x4)
        merge6 = torch.cat([up_6, x3], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, x2], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        c11 = self.conv11(c10)

        out = nn.Sigmoid()(c11)
        return out
