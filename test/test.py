import tensorflow as tf
from utility.utility import *
from PIL import Image
import numpy as np

fp = get_path('data') + '/Lina.png'
filename_queue = tf.train.string_input_producer([fp])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_png(value)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image = my_img.eval()
    Image.fromarray(np.asarray(image)).show()
    coord.request_stop()
    coord.join(threads)
