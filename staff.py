import numpy as np

class Staff(object):

	def __init__(self, max_measures, key_signature):
		self.max_measures = max_measures
		self.key_signature = key_signature
		#initialize with the 1-chord
		