import torch
import torch.nn as nn
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgba2rgb
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn import MaxUnpool2d


def conv_layer(chann_in, chann_out, kernel_size=3, padding_size=0, padding_mode='zeros', stride=1):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=kernel_size,
                  padding=padding_size, padding_mode=padding_mode, stride=stride),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            conv_layer(3, 16, 3, 1, 'reflect'),
            conv_layer(16, 16, 3, 1, 'reflect'),
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv1 = nn.Sequential(
            conv_layer(16, 32, 3, 1, 'reflect'),
            conv_layer(32, 32, 3, 1, 'reflect'),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Sequential(
            conv_layer(32, 64, 3, 1, 'reflect'),
            conv_layer(64, 64, 3, 1, 'reflect'),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv3 = nn.Sequential(
            conv_layer(64, 128, 3, 1, 'reflect'),
            conv_layer(128, 128, 3, 1, 'reflect'),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck_conv = conv_layer(128, 128, 3, 1, 'reflect')

        self.upsample0 = MaxUnpool2d(2, stride=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            conv_layer(256, 128, 3, 1, 'reflect'),
            conv_layer(128, 64, 3, 1, 'reflect'),
        )
        self.upsample1 = MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            conv_layer(128, 64, 3, 1, 'reflect'),
            conv_layer(64, 32, 3, 1, 'reflect'),
        )
        self.upsample2 = MaxUnpool2d(2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            conv_layer(64, 32, 3, 1, 'reflect'),
            conv_layer(32, 16, 3, 1, 'reflect'),
        )
        self.upsample3 = MaxUnpool2d(2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            conv_layer(32, 16, 3, 1, 'reflect'),
            nn.Conv2d(16, 1, kernel_size=3,
                      padding=1, padding_mode='reflect', stride=1),
            nn.LayerNorm((256, 256))
        )

    def forward(self, x):
        conv0 = self.enc_conv0(x)
        x, i0 = self.pool0(conv0)

        conv1 = self.enc_conv1(x)
        x, i1 = self.pool1(conv1)

        conv2 = self.enc_conv2(x)
        x, i2 = self.pool2(conv2)

        conv3 = self.enc_conv3(x)
        x, i3 = self.pool3(conv3)

        x = self.bottleneck_conv(x)

        x = self.upsample0(x, i3)
        x = torch.cat([x, conv3], dim=1)
        x = self.dec_conv0(x)

        x = self.upsample1(x, i2).to(device)
        x = torch.cat([x, conv2], dim=1)
        x = self.dec_conv1(x)

        x = self.upsample2(x, i1)
        x = torch.cat([x, conv1], dim=1)
        x = self.dec_conv2(x)

        x = self.upsample3(x, i0)
        x = torch.cat([x, conv0], dim=1)
        x = self.dec_conv3(x)
        shape = x.shape
        x = x.view(x.size(0), -1)
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x = x.view(shape)
        return x


class PlantsSemantic:
    def __init__(self, model_file):
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = None
        if torch.cuda.is_available():
            model = torch.load(model_file)
        else:
            model = torch.load(model_file, map_location=torch.device('cpu'))
        model.eval()
        self.model = model.to(self.device)

    def __open__(self, img_path):
        img = imread(img_path)
        img = self.__preprocess_image__(img, (256, 256))
        return img

    def __preprocess_image__(self, image, size):
        return np.array(resize(image, size, mode='constant') if image.shape[-1] == 3 else resize(rgba2rgb(image), size, mode='constant'), np.float32)

    def process(self, img_path, output_path=None):
        img = self.__open__(img_path)
        res = self.model(TF.to_tensor(img)[None, :, :, :].to(self.device))
        out = torch.where(res > 0.5, 1.0, 0.0).cpu().numpy()[0][0]
        if output_path:
            sav = out * 255
            imsave(output_path, sav.astype(np.uint8))
        return out
