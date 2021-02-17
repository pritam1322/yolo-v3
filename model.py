import torch
import torch.nn as nn




architecture_config = [
    (3,32,1,1),
    (3,64,2,1),
    [(1,32,1,1),(3,64,2,1), 1],
    # 64 is out_channel from previous convolution
    (3,128,2,1),
    [(1,64,2,1),(3,128,2,1), 2],
    (3,256,2,1),
    [(1,128,2,1),(3,256,2,1), 8],
    (3,512,2,1),
    [(1,256,2,1),(3,512,2,1), 4],
    (3,1024,2,1),
    [(1,512,2,1),(3,1024,2,1), 4],

]

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
'''
class Residual_block(nn.Module):
    def __init__(self, in_channels, num_repeats, **kwargs):
        super(Residual_block, self).__init__()
        self.in_channels = in_channels
        self.num_repeats = num_repeats
        self.residual = self.conv_layer(in_channels)



    def forward(self, x):
        residual = x
        out = residual(x)
        out += residual

        return out

    def conv_layer(self, in_channels):
        layers = []
        for _ in range(self.num_repeats):
            layers += [
                CNN(in_channels, in_channels//2, kernel_size =1, stride = 1, padding=1)
                ]

            layers  += [
                CNN(in_channels//2, in_channels, kernel_size =3, stride = 2, padding=1)
            ]
        return nn.Sequential(*layers)

class Conv_layers(nn.Module):
    def __init__(self, in_channels , **kwargs):
        super(Conv_layers, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config

    def forward(self):
        layers = []
        in_channels = self.in_channels
        layers += [
                    CNN(in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3] )
                ]
        in_channels = x[1]
        return nn.Sequential(*layers)

'''





class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes = 80, **kwargs):
        super(Yolov3, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.num_classes = num_classes #C
        self.darknet = self.create_conv_layers(self.architecture)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)
        #self.fcs = self.create_fcs(**kwargs)
        #self.conv1 = Conv_layers()
        #self.residual_block1 = Residual_block(in_channels)
        #self.conv2 = Conv_layers()

    def forward(self, x):
        x = self.darknet(x)
        return self.global_avg_pool(self.fcs(torch.flatten(x, dim = 1)))

    def create_conv_layers(self, architecture):
        in_channels = self.in_channels
        blocks1 = []
        blocks2 = []
        for x in self.architecture :
            if type(x) == tuple:
                 blocks1.append(
                     CNN(in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3])
                 )
                 in_channels = x[1]

            elif type(x) == list :
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    blocks2.append(
                        CNN(in_channels, conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3])
                    )
                    blocks2.append(
                        CNN(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])
                    )
                    in_channels = conv2[1]

            blocks1 += blocks2
            blocks2.clear()

        return nn.Sequential(*blocks1)


model = Yolov3()
print(model)