from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms

def preprocess(batch_size, size=(256,256)):
    transform_train = transforms.Compose([
        transforms.Resize(size=size),
        # Transpose the image from height*width*num_channels to num_channels*height*width
        # and map values from [0, 255] to [0,1]
        transforms.ToTensor(),
        # Normalize the image with mean and standard deviation calculated across all images
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=False, last_batch='discard')

    return train_data  # 我们需要的是 for i, batch in enumerate(train_data)中的batch[0](也就是图片数据，我们不需要标签数据)


def postprocess(res_images):
    # res_images : (batch_size, c, h, w)
    for i in range(len(res_images)):
        img = (res_images[i].transpose((1, 2, 0)) * nd.array([0.2023, 0.1994, 0.2010]) + nd.array(
            [0.4914, 0.4822, 0.4465])).clip(0, 1)
        res_images[i] = img.transpose((2, 0, 1))
    return res_images  # 处理完的形状依然是(batch_size, c, h, w)
