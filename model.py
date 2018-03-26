import threading, tensorflow as tf, numpy as np, sys
from tensorflow.nn import static_rnn
from tensorflow.contrib import rnn
import numpy.linalg as la
from utils import *
from music21 import chord
from music21 import key
from music21 import roman
'''
ideas: reward for creating a fugue
reward for avoiding awkward chords (iii64)

'''

class PartWriter(object):
	
	def __init__(self, sess, ignore_checkpoint, save, max_length, num_units, learning_rate, beta1, beta2):
		self.max_length = max_length
		self.chord_roll = np.zeros((4, max_length), dtype=uint8)
		self.current_index = 0
		self.num_units = num_units
		self.n_input = 4
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.n_output = 4 #output a vector of R4 that will be rounded ad. hoc to yield integers
		#todo
		
		self.build_model()
		
	def build_model(self):
		#todo
		self.chords = tf.placeholder(tf.uint8, [4, self.max_length])
		
		#need to be initialized so that the output voices are in range
		self.output_weights = tf.Variable(tf.random_normal([self.num_units, self.n_output]))
		self.output_bias = tf.Variable(tf.random_normal([self.n_output]))
		
		self.gru_cell = rnn.GRUCell(self.num_units, input_size=self.n_input, activation=swish)
		self.outputs, _ = rnn.static_rnn(self.gru_cell,chords,dtype="float32")
		self.predicted_chord = tf.matmul(self.outputs[-1], self.out_weights)+self.out_bias
		
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
		self.saver.restore(self.sess,self.get_checkpoint_file())
	
	def get_checkpoint_file(self):
		return "./{}/policy_net.ckpt".format(self.checkpoint_dir)
	
	def save_model(self):
		self.saver.save(self.sess,self.get_checkpoint_file())
		
	def train(self):
		#todo
		
	def test(self):
		#todo
	
	def check(self, last_chord, this_chord):
		#TODO: put all checks here
	
	def check_repetition(self, last_chord, this_chord):
		return not (last_chord[0]==this_chord[0] and last_chord[1]==this_chord[1] and last_chord[2]==this_chord[2] and last_chord[3]==this_chord[3])
	
	#chord = np.array
	def check_spelling(self, chord) -> bool:		
		#if duplicates exist, then the chord should be a triad
		if len(chord) != len(set(chord)):
			#must be triad to be valid
			if get_root_position_triad(chord) != None:
				return True
		else:
			#must be 7th chord
			pass #TODO
		return False
	
	#key_sig type is key.KeySignature
	def check_leading_tone_doubling(self, chord, key_sig) -> bool:
		if key_sig.mode == 'minor':
			leading_tone = (key_sig.tonic.midi - 2) % 12
			occurences = 0
			for pitch in chord:
				if pitch % 12 == leading_tone:
					occurences += 1
			if occurences > 1:
				return False
		elif key_sig.mode == 'major':
			leading_tone = (key_sig.tonic.midi - 1) % 12
			occurences = 0
			for pitch in chord:
				if pitch % 12 == leading_tone:
					occurences += 1
			if occurences > 1:
				return False
		else:
			return False #mode not recognized
		return True
		
	def check_melodic_augmented_seconds(self, last_chord, this_chord) -> bool:
		for i in range(len(last_chord)):
			if abs(last_chord[i] - this_chord[i])%12==3:
				return False
		return True
		
	def check_melodic_tritones(self, last_chord, this_chord) -> bool:
		for i in range(len(last_chord)):
			if abs(last_chord[i] - this_chord[i])%12==6:
				return False
		return True
	
	def check_voice_ranges(self, chord) -> bool:
		if chord[0] > 79 or chord[0] < 60:
			return False
		if chord[1] > 72 or chord[1] < 55:
			return False
		if chord[2] > 67 or chord[2] < 48:
			return False
		if chord[3] > 60 or chord[3] < 43:
			return False
		return True
	
	def check_spacing(self, chord) -> bool:
		if chord[0] - chord[1] > 12 or chord[1] - chord[2] > 12:
			return False
		return True
	
	def check_crossover(self, chord) -> bool:
		return chord[0] > chord[1] and chord[1] > chord[2] and chord[2] > chord[3]
		
	def check_overlap(self, last_chord, this_chord) -> bool:
		if this_chord[0] < last_chord[1] or this_chord[0] < last_chord[2] or this_chord[0] < last_chord[3]:
			return False
		if this_chord[1] > last_chord[0] or this_chord[1] < last_chord[2] or this_chord[1] < last_chord[3]:
			return False
		if this_chord[2] > last_chord[0] or this_chord[2] > last_chord[2] or this_chord[2] < last_chord[3]:
			return False
		if this_chord[3] > last_chord[0] or this_chord[3] > last_chord[2] or this_chord[3] > last_chord[3]:
			return False
		return True
	
	def check_fifths(self, last_chord, this_chord) -> bool:
		for i in range(len(last_chord)):
			for j in range(i+1, len(last_chord)):
				if (last_chord[i] - last_chord[j]) % 12 == 7 and (this_chord[i] - this_chord[j]) % 12 == 7:
					return False
		return True
	
	def check_octaves(self, last_chord, this_chord) -> bool:
		for i in range(len(last_chord)):
			for j in range(i+1, len(last_chord)):
				if (last_chord[i] - last_chord[j]) % 12 == 12 and (this_chord[i] - this_chord[j]) % 12 == 12:
					return False
		return True
	
	#TODO ONLY OCCURS IF THE SEVENTH IS MISSING THE THIRD
	def check_doubled_seventh(self, chord, key_sig) -> bool:
		if get_root_position_seventh(chord) != None:
			if key_sig.mode == 'minor':
				leading_tone = (key_sig.tonic.midi - 2) % 12
				occurences = 0
				for pitch in chord:
					if pitch % 12 == leading_tone:
						occurences += 1
				if occurences > 1:
					return False
			elif key_sig.mode == 'major':
				pass
			else:
				return False
	
	def check_cadence(self, last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig):
		return check_perfect_authentic_cadence_triad(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig)
				or check_imperfect_authentic_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig)
				or check_deceptive_cadence_triad(last_chord, this_chord, key_sig)
				or check_phyrgian_half_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig)
				or check_half_cadence_triad(this_chord_mod12, key_sig)
		
	def check_perfect_authentic_cadence_triad(self, last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig) -> bool:
		for i in range(4):
			if key_sig.mode=='major':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in I_pitches:
					return False
			elif key_sig.mode=='minor':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in i_pitches:
					return False
		
		if not is_root_position_triad(last_chord) or not is_root_position_triad(this_chord):
			return False
		
		if (this_chord[0] - key_sig.tonic.midi) % 12 is not 0:
			return False
			
		return True
	
	def check_imperfect_authentic_cadence_triad(self, last_chord_mod12, this_chord_mod12, key_sig) -> bool:
		for i in range(4):
			if key_sig.mode=='major':
				if last_chord_mod12[i] not in V_pitches or last_chord_mod12[i] not in viio_pitches or this_chord_mod12[i] not in I_pitches:
					return False
			elif key_sig.mode=='minor':
				if last_chord_mod12[i] not in V_pitches or last_chord_mod12[i] not in viio_pitches or this_chord_mod12[i] not in i_pitches:
					return False
		return True
	
	#V -> vi or VI
	def check_deceptive_cadence_triad(self, last_chord, this_chord, key_sig) -> bool:
		for i in range(4):
			if key_sig.mode=='major':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in vi_pitches:
					return False
			elif key_sig.mode=='minor':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in VI_pitches:
					return False
		
		return True
	
	#IV or iv -> I
	def check_plagal_cadence_triad(self, last_chord, this_chord, key_sig) -> bool:
		for i in range(4):
			if key_sig.mode=='major':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in IV_pitches:
					return False
			elif key_sig.mode=='minor':
				if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in iv_pitches:
					return False
		
		return True

	#vi6 -> V (only minor keys)
	def check_phyrgian_half_cadence_triad(self, last_chord_mod12, this_chord_mod12, key_sig) -> bool:
		if key_sig.mode is not 'minor':
			return False
		
		for i in range(4):
			if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in iv6_pitches:
				return False
		return True
	
	#any -> V
	def check_half_cadence_triad(self, this_chord_mod12, key_sig) -> bool:
		for i in range(4):
			if last_chord_mod12[i] not in V_pitches:
				return False
		return True
	
	def reward(self, last_chord, this_chord, last_chord_root_pos, this_chord_root_pos, key_sig):
		reward = 0
		if reward_circle_progression(last_chord_root_pos, this_chord_root_pos):
			reward += 1
		if reward_cadence(last_chord, this_chord, key_sig):
			reward += 1
		
		return reward
	
	def reward_circle_progression(self, last_chord_root_pos, this_chord_root_pos) -> bool:
		if last_chord_root_pos[2] - this_chord_root_pos[2] > 0 and (classify_triad(this_chord_root_pos) - classify_triad(last_chord_root_pos)) % 8 == 4:
			return True
		if last_chord_root_pos[2] - this_chord_root_pos[2] < 0 and (classify_triad(last_chord_root_pos) - classify_triad(this_chord_root_pos)) % 8 == 5:
			return True
		return False
		
	def reward_cadence(self, last_chord, this_chord, key_sig) -> bool:
		last_chord_mod12 = np.array([x%12 for x in last_chord])
		this_chord_mod12 = np.array([x%12 for x in this_chord])
		return check_cadence(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig)
		
	def reward_second_inversion(self, last_chord, this_chord) -> bool:
		pass #todo