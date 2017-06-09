import unittest
import tensorflow as tf
from utility.utility import *
from PIL import Image
import numpy as np


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        pass

    def test_image_load(self):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(get_path('data') + '/*.png'))
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        img = tf.image.decode_png(value)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            image = img.eval()
            # Image.fromarray(np.asarray(image)).show()
            coord.request_stop()
            coord.join(threads)
