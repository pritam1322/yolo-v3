import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNN, self).__init__()
        self.conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self.layers = nn.ModuleList()            #use ModuleList() only
        self.layers += [
            nn.Sequential(
                CNN(in_channels, in_channels//2, kernel_size=1, padding=0),
                CNN(in_channels//2, in_channels, kernel_size=3)
            )
        ]

    def forward(self, x):

        for layer in self.layers:
            x = x + layer(x)
        return x




class YOLO_v3(nn.Module):
    def __init__(self, num_classes,in_channels=3, features=32):
        super(YOLO_v3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = CNN(in_channels, features, kernel_size= 3, stride=1)   #(3,32)
        self.conv2 = CNN(features, features*2, kernel_size=3, stride=2)     #(32,64)
        self.res1 = self.make_layers(features*2, num_repeats=1)                     #(64, 32) (32, 64)
        self.conv3 = CNN(features*2, features*4, kernel_size=3, stride=2)   #(64,128)
        self.res2 = self.make_layers(features*4, num_repeats=2)
        self.conv4 = CNN(features*4, features*8, kernel_size=3, stride=2)   #(128,256)
        self.res3= self.make_layers(features*8, num_repeats=8)
        self.conv5= CNN(features*8, features*16, kernel_size=3, stride=2)   #(256,512)
        self.res4 = self.make_layers(features*16, num_repeats=8)
        self.conv6 = CNN(features*16, features*32, kernel_size=3, stride=2)
        self.res5 = self.make_layers(features*32, num_repeats=4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = self.conv6(x)
        x = self.res5(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        return self.fc(x)

    def make_layers(self, in_channels, num_repeats):
        layers = []
        for num in range(num_repeats):
            layers.append(Residual(in_channels))

        return nn.Sequential(*layers)


x = torch.randn((2,3,416,416))

model = YOLO_v3(num_classes=80)
out = model(x)
print(x.shape)
print(out.shape)