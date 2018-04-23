import threading, tensorflow as tf, numpy as np, sys
from tensorflow.contrib import rnn
import numpy.linalg as la
import random as rand
from utils import *
from rules import check
from rules import reward
from music21 import chord
from music21 import key
from music21 import roman
from music21 import pitch
import mido
import time
import math
'''
ideas: reward for creating a fugue
reward for avoiding awkward chords (iii64)

'''
'''
class RNNPartWriter(object):
	
	def __init__(self, sess, ignore_checkpoint, save, epochs, runs_per_update, max_length, num_units, learning_rate, beta1, beta2):
		self.epochs = epochs
		self.runs_per_update = runs_per_update
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
		self.chords = tf.placeholder(tf.uint8, [self.max_length,4])
		
		#need to be initialized so that the output voices are in range
		self.output_weights = tf.Variable(tf.random_normal([self.num_units, self.n_output]))
		self.output_bias = tf.Variable(tf.random_normal([self.n_output]))
		
		self.gru_cell = rnn.GRUCell(self.num_units, input_size=self.n_input, activation=swish)
		self.outputs, _ = rnn.static_rnn(self.gru_cell,chords,dtype="float32")
		self.logits = tf.matmul(self.outputs[-1], self.out_weights)+self.out_bias
		
		self.cross_entropy  =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
		
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		#summary  statistics for use with Tensorboard
		self.step_placeholder = tf.placeholder(tf.int32, shape=())
		self.errors_placeholder = tf.placeholder(tf.int32, shape=())
		
		tf.summary.scalar("steps",self.step_placeholder)
		tf.summary.scalar("errors",self.errors_placeholder)

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
		pass #todo
		
	def test(self):
		pass #todo
'''








class DQNPartWriter(object):
	def __init__(self, sess, ignore_checkpoint, save, epochs, runs_per_update, max_length, num_units, learning_rate, beta1, beta2):
		self.sess = sess
		
		self.epochs = epochs
		self.runs_per_update = runs_per_update
		self.max_length = max_length #in quarter notes
		self.n_measures = math.ceil(max_length / 4)
		
		self.chord_roll = np.zeros((max_length,4), dtype=np.uint8)
		self.last_chord_index = None
		
		self.num_units = num_units
		self.n_input = 4
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.output_dim = [4, 37] #4 voices and 37 possible outputs
		
		self.update_ops = []
	
		#todo
		
		self.build_model()
		
	def build_model(self):
		self.initializer = tf.contrib.layers.variance_scaling_initializer() #He initialization

		self.chords = tf.placeholder(tf.uint8, shape=[self.max_length, 4])
		self.last_chord_ph = tf.placeholder(tf.uint8, shape=[])
		
		#online and target q networks
		self.online_scope = "q_networks/online"
		self.target_scope = "q_networks/target"
		
		self.online_q_values = q_network(chords[self.last_chord_ph, ::], name=self.online_scope)
		self.target_q_values = q_network(chords[self.last_chord_ph, ::], name=self.target_scope)
		
		self.copier = ModelParametersCopier(online_scope,target_scope)
		
		with tf.variable_scope("train"):
			
		
			#need to be initialized so that the output voices are in range
			self.logits = tf.matmul(self.outputs[-1], self.out_weights)+self.out_bias
			
			self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
		
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		#summary  statistics for use with Tensorboard
		self.reward_placeholder = tf.placeholder(tf.int32, shape=())
		self.errors_placeholder = tf.placeholder(tf.int32, shape=())
		
		tf.summary.scalar("rewards",self.reward_placeholder)
		tf.summary.scalar("errors",self.errors_placeholder)

		self.summary = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter('summary/train', self.sess.graph)
		self.test_writer = tf.summary.FileWriter('summary/test', self.sess.graph)
		
		if not self.ignore_checkpoint:
			self.load_model()
		else:
			self.init.run()
			self.sess.run(self.update_ops)
		
	def q_network(self, past_chord, name):
		with tf.variable_scope(name) as scope:
			hidden_layers = []
			logits = []
			outputs = []
			for voice in range(self.output_dim[0]):
				hidden = tf.layers.dense(past_chord[voice], self.n_hidden, activation=swish, kernel_initializer=self.initializer)
				logit = tf.layers.dense(hidden, self.output_dim[1] ,kernel_initializer=self.initializer)
				output = tf.nn.sigmoid(logit)
				hidden_layers.append(hidden)
				logits.append(logit)
				outputs.append(output) #probabilities
			
	def copy_parameters(self, online_scope, target_scope):
		if len(self.update_ops) == 0:
			e1_params = [t for t in tf.trainable_variables() if t.name.startswith(online_scope)]
			e1_params = sorted(e1_params, key=lambda v: v.name)
			e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target_scope)]
			e2_params = sorted(e2_params, key=lambda v: v.name)

			for e1_v, e2_v in zip(e1_params, e2_params):
				op = e2_v.assign(e1_v)
				self.update_ops.append(op)
			
			sess.run(self.update_ops)
	def load_model(self):
		self.saver.restore(self.sess,self.get_checkpoint_file())
	
	def get_checkpoint_file(self):
		return "./{}/policy_net.ckpt".format(self.checkpoint_dir)
	
	def save_model(self):
		self.saver.save(self.sess,self.get_checkpoint_file())
	
	def last_chord(self):
		if self.last_chord_index is not None:
			return self.chord_roll[self.last_chord_index] #returns array with size = 4
		return None
	
	def train(self):
		if self.ignore_checkpoint:
			self.init.run()
		for iteration in range(self.epochs):
			all_rewards = [] #all sequences of raw rewards for each episode
			all_gradients = [] #gradients saved at each step of each episode
			steps = []
			for game in range(self.runs_per_update):
				print("Running game #{}".format(game))
				current_rewards = []
				current_gradients = []
				obs = self.env.reset()
				self.last_chord_index = None
				for step in range(self.max_length):
					action_val, gradients_val = self.sess.run(
							[self.action,self.gradients],
							feed_dict={self.X:obs.reshape(1,self.n_inputs)}) #one observation
					
					#action_val should be an array of size = 4
					
					#reward_val = reward(
					
					current_rewards.append(reward) #raw reward
					current_gradients.append(gradients_val) #raw grads
					if done or step==self.max_steps-1:
						print("Finished game #{} in {} steps".format(game,step+1))
						steps.append(step)
						break
				all_rewards.append(current_rewards) #adds to the history of rewards
				all_gradients.append(current_gradients) #gradient history
			
			#all games executed--time to perform policy gradient ascent
			print("Performing gradient ascent at iteration {}".format(iteration))
			all_rewards = self.discount_and_normalize_rewards(all_rewards, self.discount_rate)
			mean_reward = np.array(np.mean(
					[reward
						for rewards in all_rewards
						for reward in rewards],
					axis=0),dtype=np.float32)
			mean_steps = np.mean(steps,axis=0)
			feed_dict = {}
			for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
				#multiplication by the "action scores" obtained from discounting the future events appropriately--meaned to average the signals
				vanilla_gradient = np.mean(
					[reward*all_gradients[game_index][step][var_index] #iterates through each variable in the gradient (var_index)
						for game_index, rewards in enumerate(all_rewards)
						for step,reward in enumerate(rewards)],
					axis=0) #may be 4x4
				feed_dict[grad_placeholder] = natural_gradients
				
			self.sess.run(self.training_op,feed_dict=feed_dict)
			sum = self.sess.run(self.summary,feed_dict={self.step_placeholder:mean_steps})
			self.train_writer.add_summary(sum,iteration)
			if (iteration +1)% self.save_iterations == 0 and self.save:
				print("Saving model...")
				self.saver.save(self.sess,self.get_checkpoint_file())
		
		#finally test
		self.test()
		
	def test(self):
		pass #todo
		
	def sample_song(self):
		pass #todo... samples one song from the artificial neural network 

		
class DNNPartWriter(object):
	def __init__(self, sess, ignore_checkpoint, save, epochs, discount_rate,
						runs_per_update, max_length, num_units, learning_rate, 
						momentum, iterations_per_save, test_episodes, checkpoint_dir):
		self.sess = sess
		self.ignore_checkpoint = ignore_checkpoint
		self.save = save
		self.epochs = epochs
		self.discount_rate = discount_rate
		
		self.runs_per_update = runs_per_update
		self.max_length = max_length #in quarter notes
		self.n_measures = math.ceil(max_length / 4) # 4/4 time
		
		self.chord_roll = np.zeros((max_length,4), dtype=np.uint8)
		self.last_chord_index = None
		
		self.num_units = num_units
		self.n_input = 4
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.iterations_per_save = iterations_per_save
		self.test_episodes = test_episodes
		self.checkpoint_dir = checkpoint_dir
		self.output_dim = [4, 37] #4 voices and 37 possible outputs
		
		self.build_model()
		
	def build_model(self):
		self.initializer = tf.contrib.layers.variance_scaling_initializer() #He initialization

		self.chords = tf.placeholder(tf.uint8, shape=[self.max_length, 4])
		self.last_chord_ph = tf.placeholder(tf.float32, shape=[1, 4])
		
		hidden_layers = []
		logits = []
		outputs = []
		choices = []
		for voice in range(self.output_dim[0]):
			hidden = tf.layers.dense(self.last_chord_ph, self.num_units, activation=swish, kernel_initializer=self.initializer)
			logit = tf.layers.dense(hidden, self.output_dim[1], kernel_initializer=self.initializer)
			output = tf.nn.sigmoid(logit)
			sample = tf.multinomial(tf.log(output),num_samples=1)[:, 0] + 43 #samples chord members from probability distribution... 4x1 tensor -> rank 1 size 4 tensor.
			#returns the index of the value chosen. adds 43 because G2 is the smallest possible value
			hidden_layers.append(hidden)
			logits.append(logit)
			outputs.append(output) #probabilities
			choices.append(sample) 
		
		logits = tf.convert_to_tensor(logits, dtype=tf.float32)[:,0,:] #new tensor: shape = (4, 37)
		self.choices = tf.convert_to_tensor(choices) #new tensor: shape = (4) -- for tf.multinomial
		self.choices = tf.tile(self.choices, [1,37])
		self.choices = tf.cast(self.choices, dtype=tf.float32) #HACK: this is unnecessary for gradient injection... is there a work-around for it?
		
		self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.choices, logits=logits)
		#self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy)
		self.gradients = [grad for grad, var in grads_and_vars]
		self.gradient_placeholders = []
		grads_and_vars_feed = []
		for grad, var in grads_and_vars:
			gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
			self.gradient_placeholders.append(gradient_placeholder)
			grads_and_vars_feed.append((gradient_placeholder, var))
		self.training_op = self.optimizer.apply_gradients(grads_and_vars_feed)
		
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		#summary  statistics for use with Tensorboard
		self.loss_placeholder = tf.placeholder(tf.float32, shape=())
		self.reward_placeholder = tf.placeholder(tf.int32, shape=())
		self.errors_placeholder = tf.placeholder(tf.int32, shape=())
		
		tf.summary.scalar("loss",self.loss_placeholder)
		tf.summary.scalar("rewards",self.reward_placeholder)
		tf.summary.scalar("errors",self.errors_placeholder)

		self.summary = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter('summary/train', self.sess.graph)
		self.test_writer = tf.summary.FileWriter('summary/test', self.sess.graph)
		
		if not self.ignore_checkpoint:
			self.load_model()
		else:
			self.init.run()
	
	def load_model(self):
		self.saver.restore(self.sess,self.get_checkpoint_file())
	
	def get_checkpoint_file(self):
		return "./{}/policy_net.ckpt".format(self.checkpoint_dir)
	
	def save_model(self):
		self.saver.save(self.sess,self.get_checkpoint_file())
	
	def last_chord(self):
		if self.last_chord_index is not None:
			return self.chord_roll[self.last_chord_index] #returns array with size = 4
		return None
	
	def is_last_chord(self):
		return self.last_chord_index == self.max_length - 2
	
	def set_last_chord(self, chord):
		if self.last_chord_index == None:	
			self.last_chord_index = 0
		else:
			self.last_chord_index += 1
		self.chord_roll[self.last_chord_index] = chord
	
	def reset_system(self):
		self.last_chord_index = 0
		self.chord_roll.fill(0) #resets the chord roll
		self.key_signature = key.Key("CM")
		'''self.key_signature = key.KeySignature(rand.randint(-7,7))
		if bool(rand.getrandbits(1)):
			self.key_signature = self.key_signature.asKey(mode="major")
		else:
			self.key_signature = self.key_signature.asKey(mode="minor")'''
		#start with the I or i chord in root position with the root in the 3rd octave
		tonic = self.key_signature.tonic.midi - 12
		self.chord_roll[0] = [tonic, tonic+4 if self.key_signature=="major" else tonic+3, tonic+7, tonic+12][::-1]
	
	#rewards in the form of a 1D py array
	def discount_rewards(self, rewards, discount_rate):
		discounted_rewards = np.empty(len(rewards))
		cumulative_rewards = 0
		for step in reversed(range(len(rewards))):
			cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
			discounted_rewards[step] = cumulative_rewards
		return discounted_rewards

	#all_rewards in the form of a 2D py array
	def discount_and_normalize_rewards(self, all_rewards, discount_rate):
		all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
		flat_rewards = np.concatenate(all_discounted_rewards) #converts py list to numpy array
		reward_mean = flat_rewards.mean()
		reward_std = flat_rewards.std()
		print(reward_mean)
		print(reward_std)
		if reward_std < 0.00001:
			return reward_mean, [np.zeros(len(discounted_rewards)) for discounted_rewards in all_discounted_rewards]
		return reward_mean, [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
	
	def train(self):
		for iteration in range(self.epochs):
			all_rewards = [] #all sequences of raw rewards for each episode
			all_errors = [] 
			all_losses = []
			all_gradients = [] #gradients saved at each step of each episode
			for song in range(self.runs_per_update):
				print("Running song #{}".format(song))
				current_rewards = []
				current_errors = 0
				current_losses = []
				current_gradients = []
				self.reset_system() 
				for step in range(self.max_length):
					last_chord = self.last_chord()
					loss, new_chord, gradients_val = self.sess.run(
							[self.cross_entropy, self.choices,self.gradients],
							feed_dict={self.last_chord_ph:last_chord.reshape((1, 4))})
					new_chord = new_chord[:, 0].astype(np.uint8)
					print(last_chord)
					print(new_chord)
					#action_val should be an array of size = 4
					last_chord_root_pos = get_root_position_triad(last_chord)
					new_chord_root_pos = get_root_position_triad(new_chord)
					print(last_chord_root_pos)
					print(new_chord_root_pos)
					
					reward_val, error = reward(last_chord, new_chord, last_chord_root_pos, new_chord_root_pos, self.key_signature, self.is_last_chord())
					print("Rewards: {}".format(reward_val))
					current_rewards.append(reward_val)
					current_gradients.append(gradients_val)
					current_losses.append(loss)
					current_errors+=error
					if error:
						print("Error in song #{}... ending early".format(song))
						break
					if step==self.max_length-1:
						print("Finished song #{} with {} errors".format(song, current_errors))
						break
					
					self.set_last_chord(new_chord)
					
				all_rewards.append(current_rewards) #adds to the history of rewards
				all_gradients.append(current_gradients) #gradient history
				all_errors.append(current_errors)
				all_losses.append(current_losses)
			
			#all games executed--time to perform policy gradient ascent
			print("Performing gradient ascent at iteration {}".format(iteration))
			mean_reward, all_rewards = self.discount_and_normalize_rewards(all_rewards, self.discount_rate)
			print(all_rewards)
			
			mean_error = np.mean([error for error in all_errors])
			mean_loss = np.mean([loss for losses in all_losses for loss in losses])
			feed_dict = {}
			for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
				#multiplication by the "action scores" obtained from discounting the future events appropriately--meaned to average the signals
				vanilla_gradient = np.mean(
					[reward*all_gradients[game_index][step][var_index] #iterates through each variable in the gradient (var_index)
						for game_index, rewards in enumerate(all_rewards)
						for step,reward in enumerate(rewards)],
					axis=0) #in contrast to natural policy gradients, which would be more efficient #HACK
				feed_dict[grad_placeholder] = vanilla_gradient
				
			self.sess.run(self.training_op,feed_dict=feed_dict)
			sum = self.sess.run(self.summary,feed_dict={self.loss_placeholder:mean_loss, self.reward_placeholder:mean_reward, self.errors_placeholder:mean_error})
			self.train_writer.add_summary(sum,iteration)
			if (iteration +1)% self.iterations_per_save == 0 and self.save:
				print("Saving model...")
				self.save_model()
		
		#finally test
		self.test()
		
	def test(self):
		with mido.open_input() as port:
			for test in range(self.test_episodes):
				song = self.sample_song()
				for pitches in song:
					for pitch in pitches:
						msg = mido.Message('note_on', note = pitch, velocity=127)
						port.send(msg)
					time.sleep(1) # 60 bpm
		
	#only perform when not training
	def sample_song(self):
		self.reset_system()
		for step in range(self.max_length):
			last_chord = self.last_chord()
			new_chord, gradients_val = self.sess.run(
					[self.choices,self.gradients],
					feed_dict={self.last_chord_ph:last_chord.reshape((1, 4))})
			new_chord = new_chord[:, 0]
			set_last_chord(new_chord)
		return np.copy(self.chord_roll)

class AlgorithmicPartWriter(object):
	def __init__(self, max_length, episodes):
		self.max_length = max_length
		self.episodes = episodes
		
		self.n_measures = math.ceil(max_length / 4) # 4/4 time
		self.pitch_roll = np.empty((max_length,4), dtype=object) #use Note music21 class
		self.chord_roll = np.empty((max_length), dtype=object)
		self.last_pitches_index = None
		
	def generate_chords(self):
		self.chord_roll.fill(None)
		for i in range(self.max_length):
			chord_roll[i] = self.generate_chord()
		
	def generate_chord(self):
		if bool(rand.getrandbits(1)): #conditional chord generation
			return roman.RomanNumeral(rand.randint(1,7), self.key_signature)
		else: #use circle of fifths to generate the chord
			return roman.RomanNumeral((chords[i-1].scaleDegree+4)%8, self.key_signature)
		
	def next_chord(self, chord_index): #called if the current chord cannot satisfy the rules of part-writing
		chord = self.chord_roll[chord_index]
		if chord.inversion()<2:
			chord.inversion(chord.inversion()+1)
		else:
			self.chord_roll[chord_index] = self.generate_chord() #try a different chord
	
	def step(self, next_index): #next is of type RomanNumeral
		chord = self.chord_roll[next_index]
		possibilities = self.recursive_substep(3, chord, np.zeros((4), dtype=np.uint8))
		while len(possibilities)==0:
			next_chord = next_chord(next_index)
		
		return possibilities[0]
	
	def recursive_substep(self, voice_number, chord, pitches):
		if voice_number==-1:
			return pitches
		pitch_set = []
		for pitch in chord.pitches():
			midi = pitch.midi - 24 if voice_number==3 else pitch.midi - 12*(pitch.octave + pitches[voice_number+1].octave)
			while midi < 79:
				copy = pitches.copy()
				copy[voice_number] = midi
				pitch_set.append(recursive_substep(voice_number - 1, copy, pitches))
				midi += 12
		
		result = []
		for combination in pitch_set:
			if(check(combination)==17):
				result.append(combination)
		
		return result

	def last_chord(self):
		if self.last_pitches_index is not None:
			return self.pitch_roll[self.last_pitches_index] #returns array with size = 4
		return None
	
	def is_last_chord(self):
		return self.last_pitches_index == self.max_length - 2
	
	def set_last_chord(self, chord):
		if self.last_pitches_index == None:	
			self.last_pitches_index = 0
		else:
			self.last_pitches_index += 1
		self.pitch_roll[self.last_pitches_index] = chord
	
	def reset_system(self):
		self.last_pitches_index = 0
		self.pitch_roll.fill(None)
		
		#get random key
		self.key_signature = key.KeySignature(rand.randint(-7,7))
		if bool(rand.getrandbits(1)):
			self.key_signature = self.key_signature.asKey(mode="major")
		else:
			self.key_signature = self.key_signature.asKey(mode="minor")
		
		#start with the I or i chord in root position with the root in the 3rd octave
		tonic = self.key_signature.tonic
		tonic.octave = 3
		
		self.generate_chords()
		self.chord_roll[0] = roman.RomanNumeral(1, self.key_signature)
		
		self.pitch_roll[0][0] = pitch.Pitch(tonic.ps+12)
		self.pitch_roll[0][1] = pitch.Pitch(tonic.ps+7)
		self.pitch_roll[0][2] = pitch.Pitch(tonic.ps+4) if self.key_signature=="major" else pitch.Pitch(tonic.ps+3)
		self.pitch_roll[0][3] = pitch.Pitch(tonic.ps)
		
		
	
	def run(self):
		for episode in range(self.episodes):
			step_index = 1
			while step_index < max_length:
				pitches = self.step(step_index)
				self.pitch_roll[step_index] = pitches
			
			