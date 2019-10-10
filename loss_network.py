from mxnet.gluon import model_zoo, nn
from mxnet import nd

style_layers = [3, 8, 15, 22]
content_layers = [15]


def get_loss_network():
    vgg16 = model_zoo.vision.vgg16(pretrained=True)
    net = nn.Sequential()
    for i in range(max(style_layers + content_layers) + 1):
        net.add(vgg16.features[i])
    return net


def extract_features(images, net):
    contents_features = []
    styles_features = []
    for i in range(len(net)):
        images = net[i](images)
        if i in content_layers:
            contents_features.append(images)
        if i in style_layers:
            styles_features.append(images)
    return contents_features, styles_features  #(L, batch_num, c, h, w)


def content_loss(c_f_h, c_f):  # c_f_h为从合成图像中提取出来的内容特征， c_f为从内容图像中提取出来的内容特征
    # 形状都为： (batch_num, c, h, w)
    return (c_f_h - c_f).square().mean()# *c_f.shape[0]


def gram(X):
    batch_num, num_channels, n = X.shape[0], X.shape[1], X.size // (X.shape[0] * X.shape[1])
    X = X.reshape(shape=(batch_num, num_channels, n))  # 形状: (batch_num, c, h*w)
    res = []
    for x in X:
        res.append(nd.dot(x, x.T)/x.size)
    return res


def style_loss(s_f_h, s_f_gram):
    # 形状： s_f_h:(batch_num, c, h, w); s_f_gram: (1, c,c)
    s_f_h_gram = gram(s_f_h)  # (batch_num, c, c)
    return (s_f_h_gram - s_f_gram).square().sum() / s_f_h.shape[0]


def tv_loss(y_h):
    return 0.5 * ((y_h[:, :, 1:, :] - y_h[:, :, :-1, :]).abs().mean() +
                  (y_h[:, :, :, 1:] - y_h[:, :, :, :-1]).abs().mean())


def compute_loss(res_img, weights, contents_features_h, styles_features_h, contents_features, styles_features_gram):
    content_weight, style_weight, tv_weight = weights
    contents_l = [content_loss(c_f_h, c_f) * content_weight
                  for c_f_h, c_f in zip(contents_features_h, contents_features)]
    contents_l = nd.add_n(*contents_l).asscalar()
    styles_l = [style_loss(s_f_h, s_f_gram) * style_weight
                for s_f_h, s_f_gram in zip(styles_features_h, styles_features_gram)]
    styles_l = nd.add_n(*styles_l).asscalar()
    tv_l = (tv_loss(res_img) * tv_weight).asscalar()

    total_l = contents_l + styles_l + tv_l

    return total_l, contents_l, styles_l, tv_l
    # 除以了batch_num,所以后面trainner.step(num), num = 1







