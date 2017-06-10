import tensorflow as tf, numpy as np
from tqdm import trange
from logic_unit import *

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def network():
    x = tf.placeholder(tf.float32, [None, 8])
    y = AND(x, [8])
    return x, y


x, y = network()
y_ = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_mean((y_ - y) ** 2)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

for t in trange(10000):
    batch_x = np.array([np.random.randint(2, size=8) for _ in range(18)] + [[1, 1, 1, 1, 1, 1, 1, 1] for _ in range(14)])
    batch_y_ = np.array([[1] if sum(batch_x[_]) == 8 else [0] for _ in range(18)] + [[1] for _ in range(14)])
    l, to = sess.run([loss, train_op], feed_dict={x: batch_x, y_: batch_y_})
    # print(l)
result = sess.run(y, feed_dict={x: [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]]})
print(result)
