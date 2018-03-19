import numpy as np

class Chord(object):
	def __init__(self, members)
	'''
	def __init__(self, key_signature, scale_degree = 1, inversion = 0):
	self.key_signature = key_signature
	self.scale_degree = scale_degree
	self.inversion = inversion
	'''

class ChordType(Enum):
	#number of members, doubling
	TRIAD_DOUBLED_ROOT = (3, 0)
	TRIAD_DOUBLED_THIRD = (3, 1)
	TRIAD_DOUBLED_FIFTH = (3, 2)
	SEVENTH_CHORD = 4
	SECONDARY_CHORD