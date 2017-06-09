import tensorflow as tf
from utility.utility import get_path


def foo(name, sw):
    w = graph.get_tensor_by_name(name)
    value = sess.run(w)
    sum_w = sess.run(tf.summary.histogram(name, w))
    sw.add_summary(sum_w, 1000)
    return value


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(get_path("data") + "/tensorpack/" + "graph-0502-182240.meta")
    saver.restore(sess, tf.train.latest_checkpoint(get_path("data") + "/tensorpack/"))
    graph = tf.get_default_graph()
    train_var = tf.trainable_variables()
    summary_writer = tf.summary.FileWriter(get_path("tmp/tensorpack"), graph)
    node_name = [n.name for n in graph.as_graph_def().node]
    weights = [foo(v.name, summary_writer) for v in train_var if 'W' in v.name]
    summary_writer.flush()
    sess.run(tf.global_variables_initializer())
    weights_ = [foo(v.name, summary_writer) for v in train_var if 'W' in v.name]

tf.reset_default_graph()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(get_path("data") + "/badend/" + "model.ckpt-3560000.meta")
    saver.restore(sess, tf.train.latest_checkpoint(get_path("data") + "/badend/"))
    graph = tf.get_default_graph()
    train_var = tf.trainable_variables()
    summary_writer = tf.summary.FileWriter(get_path("tmp/badend"), graph)
    [foo(v.name, summary_writer) for v in train_var if 'w' in v.name]
    summary_writer.flush()
