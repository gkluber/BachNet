import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from music21 import key
import math

#MATH AND ALGORITHM UTILS
'''
Reference: 
Searching for Activation Functions (2017) 
https://arxiv.org/abs/1710.05941
'''
#let x be tf.Variable
def swish(x, name=None):
	return tf.multiply(x, tf.sigmoid(x))

#MUSIC UTILS
V_pitches = np.array([2,11,7]) #14, 11, 7 mod 12
I_pitches = np.array([7,4,0]) # 7, 4, 0 mod 12
i_pitches = np.array([7,3,0]) # 7, 3, 0 mod 12
iv6_pitches = np.array([5,0,8]) # 17, 12, 8 mod 12
iv_pitches = np.array([0,8,5]) #used with minor
IV_pitches = np.array([0,9,5]) #used with major
VI_pitches = np.array([3,0,8]) #used with minor
vi_pitches = np.array([4,0,9]) #used with major
viio_pitches = np.array([5,2,11])

def invert(chord):
	return 

def remove_doubling(pitches):
	for i in range(len(pitches)):
		for j in range(i+1,len(pitches)):
			if pitches[i]%12 == pitches[j]%12:
				return np.delete(pitches, i)
	return pitches

#returns None if the set of pitches is not a valid triad
def get_root_position_triad(pitches):
	if pitches.size > 3:
		pitches = remove_doubling(pitches)
	if is_root_position_triad(pitches):
		return pitches
	elif is_first_inversion_triad(pitches):
		pitches = np.array(pitches, copy=True)
		while pitches[0] > pitches[2]:
			pitches[0] -= 12
		return np.array([pitches[1], pitches[2], pitches[0]])
	elif is_second_inversion_triad(pitches):
		pitches = np.array(pitches, copy=True)
		while pitches[0] > pitches[2]:
			pitches[0] -= 12
		while pitches[1] > pitches[0]:
			pitches[1] -= 12
		return np.array([pitches[2], pitches[0],pitches[1]])
	return np.array([None])

def get_root_position_seventh(pitches):
	pass #TODO
	
def is_root_position_seventh(pitches):
	fifthToSeventh = pitches[0] - pitches[1]
	return is_root_position_triad(pitches[1::]) and (fifthToSeventh in (3,4))

def is_first_inversion_seventh(pitches):
	pass #todo
	
def is_second_inversion_seventh(pitches):
	pass #todo

def is_third_inversion_seventh(pitches):
	pass #todo
	
def is_root_position_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	if len(pitches) < 3:
		return False
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (3,4)) and (thirdToFifth % 12 in (3,4))

def is_first_inversion_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	if len(pitches) < 3:
		return False
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (3,4)) and (thirdToFifth % 12 in (4,5,6)) and (bassToFifth % 12 in (8,9))

def is_second_inversion_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	if len(pitches) < 3:
		return False
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (4,5,6)) and (thirdToFifth % 12 in (3,4)) and (bassToFifth % 12 in (8,9))
	
#pitches must be in root position
def classify_triad(pitches, key_sig) -> int:
	degree = (pitches[2]-key_sig.tonic.midi)%12 #gets the number of half steps from tonic to root
	if degree == 0 or degree == 1:
		return 1
	elif degree==2:
		return 2
	elif degree==3:
		if key_sig.mode=='minor':
			return 3
		elif key_sig.mode=='major':
			return 2
	elif degree==4:
		return 3
	elif degree==5 or degree==6:
		return 4
	elif degree==7:
		return 5
	elif degree==8:
		if key_sig.mode=='minor':
			return 6
		elif key_sig.mode=='major':
			return 5
	elif degree==9:
		return 6
	elif degree==10:
		if key_sig.mode=='minor':
			return 7
		elif key_sig.mode=='major':
			return 6
	elif degree==11:
		return 7
	
	return 0 #error