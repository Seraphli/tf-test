import unittest
import tensorflow as tf
from utility.utility import get_path


class TestSessionSaveLoad(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        pass

    def test_sl(self):
        # Save
        v_s = tf.Variable(1, name="v")
        v_add = v_s.assign_add(2)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            sess.run(init_op)
            sess.run(v_add)
            sess.run(v_add)
            self.assertEqual(5, sess.run(v_s))
            save_path = saver.save(sess, get_path("tmp") + "/sl.ckpt")
        tf.reset_default_graph()

        # Load
        v_l = tf.Variable(0, name="v")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            # Restore variables from disk.
            saver.restore(sess, get_path("tmp") + "/sl.ckpt")
            self.assertEqual(5, sess.run(v_l))

    def test_sl_g(self):
        # Save
        v_s = tf.get_variable("v", [], tf.int64, tf.constant_initializer(5))
        v_add = tf.assign(v_s, v_s + 3)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            sess.run(init_op)
            sess.run(v_add)
            sess.run(v_add)
            self.assertEqual(11, sess.run(v_s))
            save_path = saver.save(sess, get_path("tmp") + "/sl_g.ckpt")
        tf.reset_default_graph()

        # Load
        v_l = tf.get_variable("v", [], tf.int64, tf.zeros_initializer())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            # Restore variables from disk.
            saver.restore(sess, get_path("tmp") + "/sl_g.ckpt")
            self.assertEqual(11, sess.run(v_l))

    def test_step_sl(self):
        # Save
        step_count = 0
        step_count_tensor = tf.get_variable('step_count', [], tf.int64, tf.zeros_initializer())
        step_count_inc = tf.assign(step_count_tensor, step_count_tensor + 1)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            sess.run(init_op)
            sess.run(step_count_inc)
            step_count = sess.run(step_count_tensor)
            self.assertEqual(1, step_count)
            for i in range(100):
                sess.run(step_count_inc)
            step_count = sess.run(step_count_tensor)
            self.assertEqual(101, step_count)
            save_path = saver.save(sess, get_path("tmp") + "/step_sl.ckpt")
        tf.reset_default_graph()

        # Load
        step_count_tensor = tf.get_variable('step_count', [], tf.int64, tf.zeros_initializer())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            saver.restore(sess, get_path("tmp") + "/step_sl.ckpt")
            step_count = sess.run(step_count_tensor)
            self.assertEqual(101, step_count)

    def test_epsilon_sl(self):
        # Save
        import part.epsilon as e
        epsilon = e.LinearAnnealEpsilon(1.0, 0.1, 100)
        _epsilon = 0.0
        step_count = 0
        step_count_tensor = tf.get_variable('step_count', [], tf.int64, tf.zeros_initializer())
        step_count_inc = tf.assign(step_count_tensor, step_count_tensor + 1)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            sess.run(init_op)
            step_count = sess.run(step_count_tensor)
            _epsilon = epsilon.run_step(step_count)
            self.assertEqual(1.0, _epsilon)
            for i in range(80):
                sess.run(step_count_inc)
            step_count = sess.run(step_count_tensor)
            _epsilon = epsilon.run_step(step_count)
            self.assertGreater(1.0, _epsilon)
            self.assertLess(0.1, _epsilon)
            save_path = saver.save(sess, get_path("tmp") + "/epsilon_sl.ckpt")
        tf.reset_default_graph()

        # Load
        epsilon = e.LinearAnnealEpsilon(1.0, 0.1, 100)
        step_count_tensor = tf.get_variable('step_count', [], tf.int64, tf.zeros_initializer())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.graph.finalize()
            saver.restore(sess, get_path("tmp") + "/epsilon_sl.ckpt")
            step_count = sess.run(step_count_tensor)
            self.assertEqual(_epsilon, epsilon.run_step(step_count))
