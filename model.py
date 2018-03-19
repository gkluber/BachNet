import threading, tensorflow as tf, numpy as np, sys
import numpy.linalg as la
from utils import get_normalized_pitches
'''
ideas: reward for creating a fugue
reward for avoiding awkward chords (iii64)

'''

class PartWriter(object):
	
	def __init__(self, max_length):
		self.max_length = max_length
		#todo
		
		self.build_model()
		
	def build_model(self):
		#todo
		
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		#summary  statistics for use with Tensorboard
		self.step_placeholder = tf.placeholder(tf.int32, shape=())
		self.errors_placeholder = tf.placeholder(tf.int32, shape=())
		#self.baseline_placeholder = tf.placeholder(tf.float32, shape=())
		tf.summary.scalar("steps",self.step_placeholder)
		tf.summary.scalar("errors",self.errors_placeholder)
		#tf.summary.scalar("baseline",self.baseline_placeholder)
		self.summary = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter('summary/train', self.sess.graph)
		self.test_writer = tf.summary.FileWriter('summary/test', self.sess.graph)
		
		if not self.ignore_checkpoint:
			self.load_model()
		
	def load_model(self):
		#todo
		
	def save_model(self):
		#todo
		
	def train(self):
		#todo
		
	def test(self):
		#todo
	
	def check(self, last_chord, this_chord):
		#TODO: put all checks here
	
	#chord = np.array
	def check_spelling(self, chord):
		copy = np.fromiter((lambda x: x % 12 for pitch in chord), chord.dtype, count=len(chord)) #applies mod 12 to all chord members
		copy.sort()
		
		#if duplicates exist, then the chord should be a triad
		if len(copy) != len(set(copy)):
			#triad
			
		else:
			#7th chord
		
	def check_leading_tone_doubling(self, chord):
		#todo
		
	def check_melodic_augmented_seconds(self, last_chord, this_chord):
		#todo
		
	def check_melodic_tritones(self, last_chord, this_chord):
		#todo
	
	def check_voice_ranges(self, chord):
		if chord[0] > 79 or chord[0] < 60:
			return False
		if chord[1] > 72 or chord[1] < 55:
			return False
		if chord[2] > 67 or chord[2] < 48:
			return False
		if chord[3] > 60 or chord[3] < 43:
			return False
		return True
	
	def check_cadence(self, last_chord, this_chord):
		#todo
	
	def check_spacing(self, chord):
		if chord[0] - chord[1] > 12 or chord[1] - chord[2] > 12:
			return False
		return True
	
	def check_crossover(self, chord):
		return chord[0] > chord[1] and chord[1] > chord[2] and chord[2] > chord[3]
		
	def check_overlap(self, last_chord, this_chord):
		if this_chord[0] < last_chord[1] or this_chord[0] < last_chord[2] or this_chord[0] < last_chord[3]:
			return False
		if this_chord[1] > last_chord[0] or this_chord[1] < last_chord[2] or this_chord[1] < last_chord[3]:
			return False
		if this_chord[2] > last_chord[0] or this_chord[2] > last_chord[2] or this_chord[2] < last_chord[3]:
			return False
		if this_chord[3] > last_chord[0] or this_chord[3] > last_chord[2] or this_chord[3] > last_chord[3]:
			return False
		return True
	
	def check_parallel_fifths(self, last_chord, this_chord):
		#todo
	
	def check_parallel_octaves(self, last_chord, this_chord):
		#todo
	
	def check_contrary_fifths(self, last_chord, this_chord):
		#todo
		
	def check_contrary_octaves(self, last_chord, this_chord):
		#todo
	
	def 