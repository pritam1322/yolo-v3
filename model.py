import torch
import torch.nn as nn

"""
Convolution Block : {
            conv
            batchnorm
            leakyrelu
}
"""
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


#Residual block : {  conv_block --> conv_block ---> residual }

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()

        self.conv1 = CNN(in_channels, in_channels//2, kernel_size=1, stride=1)
        self.conv2 = CNN(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out



class detection_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(detection_block, self).__init__()
        self.detect = nn.Sequential(
            CNN(in_channels, out_channels, kernel_size=1, stride=1),
            CNN(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1),
            CNN(out_channels*2, out_channels, kernel_size=1),
            CNN(out_channels, out_channels*2, kernel_size=3, padding=1),
            CNN(out_channels*2, out_channels, kernel_size=1),


        )

    def forward(self, x):
        return self.detect(x)

class scale(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(scale, self).__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CNN(in_channels // 2, in_channels, kernel_size=3, padding=1),
            CNN(in_channels, 3*(4 + 1 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5)
                )

class yolo_v3(nn.Module):
    def __init__(self, in_channels, features):
        super(yolo_v3, self).__init__()
        self.in_channels = in_channels
        self.conv1 = CNN(in_channels, features, kernel_size=3, stride=1, padding=1)  # (3,32)
        self.conv2 = CNN(features, features * 2, kernel_size=3, stride=2, padding=1)  # (32,64)
        self.res1 = self.make_layers(features * 2, num_repeats=1)  # (64, 32) (32, 64)
        self.conv3 = CNN(features * 2, features * 4, kernel_size=3, stride=2, padding=1)  # (64,128)
        self.res2 = self.make_layers(features * 4, num_repeats=2)
        self.conv4 = CNN(features * 4, features * 8, kernel_size=3, stride=2, padding=1)  # (128,256)
        self.res3 = self.make_layers(features * 8, num_repeats=8)
        self.conv5 = CNN(features * 8, features * 16, kernel_size=3, stride=2, padding=1)  # (256,512)
        self.res4 = self.make_layers(features * 16, num_repeats=8)
        self.conv6 = CNN(features * 16, features * 32, kernel_size=3, stride=2, padding=1)
        self.res5 = self.make_layers(features * 32, num_repeats=4)

        # detection for 13 x 13
        self.scale1 = scale(features*32, num_classes=80)
        self.detect_13x13 = detection_block(features*32, features*16)

        # detection for 26 x 26
        self.conv7 = CNN(features*16, features*8, kernel_size=1, stride=1)
        self.upsamp = nn.Upsample(scale_factor=2)

        self.detect_26x26 = detection_block(features*8*3, features*8)
        self.scale2 = scale(features*16, num_classes=80)
        #detection for 52 x 52
        self.conv8 = CNN(features*8, features*4, kernel_size=1, stride=1)
        self.detect_52x52 = detection_block(features*4*3, features*4)
        self.scale3 = scale(features*8, num_classes=80)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        d0 = self.res3(x)    #detection layer 52 x52
        x = self.conv5(d0)
        d1 = self.res4(x)    #detection layer 26 x 26
        x = self.conv6(d1)
        d2 = self.res5(x)    #detection layer 13x13

        d2 = self.detect_13x13(d2)
        d2_s = self.scale1(d2)


        d1_u = self.conv7(d2)
        d1_u = self.upsamp(d1_u)
        d1 = torch.cat([d1, d1_u], dim=1)
        d1 = self.detect_26x26(d1)
        d1_s = self.scale2(d1)

        d0_u = self.conv8(d1)
        d0_u = self.upsamp(d0_u)
        d0 = torch.cat([d0, d0_u], dim=1)
        d0 = self.detect_52x52(d0)
        d0_s = self.scale3(d0)


        return d0_s, d1_s, d2_s
    def make_layers(self, in_channels, num_repeats):
        layers = []
        for num in range(num_repeats):
            layers.append(residual_block(in_channels))

        return nn.Sequential(*layers)


x = torch.randn((2, 3, 416, 416))
model = yolo_v3(in_channels=3, features=32)
pred = model(x)
print(pred)



"""
class yolo_v3_detection(nn.Module):
    def __init__(self):
        super(yolo_v3_detection, self).__init__()

        #for 13 x 13
        
        model = yolo_v3(in_channels=3, features=32)
        small,medium,large = model(x)
"""

        
