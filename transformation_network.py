from mxnet import nd
from mxnet.gluon import nn


class ResidualBlock(nn.Block):
    def __init__(self, num_channels=128, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=1)  # 大小不变
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=1)  # 大小不变
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return Y + X  # 大小不变，直接相加


def get_transform_network():
    net = nn.Sequential()
    net.add(nn.Conv2D(32, kernel_size=9, strides=1, padding=4), nn.InstanceNorm(), nn.Activation('relu'),
            nn.Conv2D(64, kernel_size=3, strides=2, padding=1), nn.InstanceNorm(), nn.Activation('relu'),
            nn.Conv2D(128, kernel_size=3, strides=2, padding=1), nn.InstanceNorm(), nn.Activation('relu'),
            ResidualBlock(),  # 五个ResidualBlock的通道数和大小和他们的输入一样
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            nn.Conv2DTranspose(64, kernel_size=3, strides=2, padding=1, output_padding=1), nn.InstanceNorm(), nn.Activation('relu'),
            nn.Conv2DTranspose(32, kernel_size=3, strides=2, padding=1, output_padding=1), nn.InstanceNorm(), nn.Activation('relu'),
            nn.Conv2D(3, kernel_size=9, strides=1, padding=4), nn.InstanceNorm(), nn.Activation('tanh'))
    # 图像输入到CNN之前，先进行标准化， 使其范围在0-1之间，
    return net

"""
img = image.imread('star.jpg')
plt.imshow(img.asnumpy())
plt.show()
image_shape = [256, 256]
img = image.imresize(img, *image_shape)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform(img)
img = img.expand_dims(axis=0)
print("img:", img)

net = get_transform_network()
net.initialize()
for i in range(len(net)):
    img = net[i](img)
    print("net[%d] shape:"%i, img.shape)

img = img[0]
img = (img.transpose((1, 2, 0)) * nd.array([0.229, 0.224, 0.225]) + nd.array([0.485, 0.456, 0.406])).clip(0, 1)
print(img)
plt.imshow(img.asnumpy())
plt.show()
"""

