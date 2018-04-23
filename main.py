import os, numpy as np, tensorflow as tf
from model import DNNPartWriter

flags = tf.app.flags
flags.DEFINE_boolean("train",True,"If True, then train the model with the given number of epochs")
flags.DEFINE_boolean("ignore_checkpoint", True, "If True, then ignore previous checkpoints")
flags.DEFINE_boolean("save", True, "If True, then save the model")
flags.DEFINE_float("learning_rate", 0.003, "Learning rate of the momentum optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum of Nesterov optimizer")
flags.DEFINE_float("discount_rate", 0.95, "Discount rate of the policy gradient algorithm")
flags.DEFINE_integer("epochs", 2000, "Epochs to train on")
flags.DEFINE_integer("num_units", 40, "Number of neurons in the hidden layer")
flags.DEFINE_integer("max_length", 20, "Maximum number of chords per episode")
flags.DEFINE_integer("runs_per_update", 10, "Number of songs per iteration")
flags.DEFINE_integer("iterations_per_save", 50, "Number of iterations per save")
flags.DEFINE_integer("test_episodes", 5, "Number of games to test with")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory that checkpoints will be saved in and loaded from")
FLAGS = flags.FLAGS

def main(_):
	print(FLAGS.__flags)
	
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	
	with tf.Session() as sess:
		'''model = DNNPartWriter(sess, FLAGS.ignore_checkpoint, FLAGS.save, FLAGS.epochs, FLAGS.discount_rate,
							FLAGS.runs_per_update, FLAGS.max_length, FLAGS.num_units, FLAGS.learning_rate,
							FLAGS.momentum, FLAGS.iterations_per_save, FLAGS.test_episodes, FLAGS.checkpoint_dir)
		if FLAGS.train:
			model.train()
		else:
			model.load_model()
			model.test()
		'''
		model = AlgorithmicPartWriter(FLAGS.max_length, FLAGS.test_episodes)
		
		model.run()

if __name__ == '__main__':
	tf.app.run()