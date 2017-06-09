import unittest
import tensorflow as tf
from utility.utility import get_path
import utility.tf_layer as layer
import numpy as np
from PIL import Image


class TestNetwork(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        pass

    def test_training(self):
        # Setup network
        w_init, b_init = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        def build_network(x, c_names):
            w, b, c = layer.conv_layer(x, [8, 8, 1, 32], [32], w_init, b_init, 4, 'c', c_names)
            fl = tf.contrib.layers.flatten(c)
            fl_size = fl.shape[1].value
            fc = layer.fc_layer(fl, [fl_size, 1], [1], w_init, b_init, 'fc', c_names)
            return fc, w, c

        x = tf.placeholder(tf.float32, [None, 84, 84, 1], name='x')
        y = tf.placeholder(tf.float32, [None, ], name='y')

        with tf.variable_scope('network'):
            c_names = ['net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            net, w_l, c_l = build_network(x, c_names)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(y, net), name='MSE')
        with tf.variable_scope('train'):
            train_op = tf.train.RMSPropOptimizer(0.01, momentum=0.95).minimize(loss)

        out_c = tf.image.convert_image_dtype(tf.transpose(c_l, [3, 1, 2, 0]), tf.uint8)

        # Read Image
        image_path = get_path('data') + '/Lina.png'
        filename_queue = tf.train.string_input_producer([image_path])
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        img = tf.image.decode_png(value, channels=1)
        true_img = tf.reshape(tf.image.convert_image_dtype(img, tf.float32), [84, 84, 1])
        true_img_ = tf.reshape(true_img, [1, 84, 84, 1])
        rand_img = tf.random_uniform([6, 84, 84, 1])
        true_imgs = tf.stack([true_img] * 26)
        images = tf.concat([true_imgs, rand_img], 0)
        data = [images, tf.stack([1.] * 26 + [0.] * 6)]
        image_batch, label_batch = tf.train.shuffle_batch(data, batch_size=32, capacity=32, min_after_dequeue=0,
                                                          enqueue_many=True)

        # Session
        prev_loss = 0
        count = 0
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            while True:
                x_ = image_batch.eval()
                y_ = label_batch.eval()
                _, _loss = sess.run([train_op, loss], feed_dict={x: x_, y: y_})
                if abs(prev_loss - _loss) < 0.001:
                    if count < 25:
                        count += 1
                    else:
                        break
                else:
                    count = 0
                prev_loss = _loss
            coord.request_stop()
            coord.join(threads)

            img = np.reshape(rand_img.eval()[0], [1, 84, 84, 1])
            np.savetxt(get_path('tmp') + '/img.txt', np.reshape(img, [84, 84]), fmt='%.6f')
            pred1 = sess.run(net, feed_dict={x: img})
            print(pred1)
            img = true_img_.eval()
            pred2 = sess.run(net, feed_dict={x: img})
            print(pred2)
            _w = sess.run(w_l, feed_dict={x: img})
            _w = np.reshape(_w, [8, 8, 32])
            np.savetxt(get_path('tmp') + '/w_out.txt', _w[0], fmt='%.6f')
            _c = sess.run(c_l, feed_dict={x: img})
            _c = np.reshape(_c, [21, 21, 32])
            np.savetxt(get_path('tmp') + '/c_out.txt', _c[0], fmt='%.6f')
            _out_c = sess.run(out_c, feed_dict={x: img})
            # for i in range(32):
            #     _img = _out_c[i, :, :, :]
            #     result = Image.fromarray(np.asarray(_img))
            #     result.save(get_path('tmp') + '/%d.jpg' % i)
