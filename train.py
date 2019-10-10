from mxnet import nd, autograd
from mxnet.gluon import Trainer
import time
import loss_network


def optimize(transform_net, loss_net, contents_images, style_img, weights, epochs):
    # contents_images: (num_iter, batchsize, c, h, w)
    # style_img: (batchsize=1, c, h, w)
    # weights = [content_weight, style_weight, tv_weight]
    trainer = Trainer(transform_net.collect_params(), 'adam', {"learning_rate": 0.001})
    _, styles_features = loss_network.extract_features(style_img, loss_net)
    styles_features_gram = loss_network.gram(styles_features)
    min_loss = 0.0
    for i in range(epochs):
        start = time.time()
        num = 0
        total_l_sum, contents_l_sum, styles_l_sum, tv_l_sum = 0.0, 0.0, 0.0, 0.0
        for contents_img in contents_images:
            contents_features, _ = loss_network.extract_features(contents_img[0], loss_net)
            with autograd.record():
                res_img = transform_net(contents_img[0])
                contents_features_h, style_features_h = loss_network.extract_features(res_img, loss_net)
                total_l, contents_l, styles_l, tv_l = loss_network.compute_loss(res_img, weights,
                                                                                contents_features_h, style_features_h,
                                                                                contents_features, styles_features_gram)
            total_l.backward()
            trainer.step(1)

            total_l_sum += total_l
            contents_l_sum += contents_l
            styles_l_sum += styles_l
            tv_l_sum += tv_l
            num += 1

        if total_l < min_loss:  # 目前： 只要得到的结果更好就保存
            min_loss = total_l
            transform_net.save_params("best_params_%d" % (i + 1))
        if (i + 1) % 50 == 0:
            print('epoch %3d, total loss %.2f, content loss %.2f, style loss %.2f, TV loss %.2f, %.2f sec'
                  % (i + 1, total_l_sum/num, contents_l_sum/num, styles_l_sum/num, tv_l_sum/num, time.time() - start))

