import numpy as np
from enum import Enum
from abc import ABC
from utils import get_normalized_pitches

class Cadence(ABC):
	
	def __init__(self):
		super().__init__()
		
	#returns name as string
	@abstractmethod
	def get_name(self):
		pass
	
	#returns true if the two chords meet the predicate for the cadence
	@abstractmethod
	def check(self, penultimate_chord, ultimate_chord):
		pass

def PerfectAuthenticCadence(Cadence):
	def __init__(self):
		super().__init__()
	
	def get_name(self):
		return "Perfect Authentic Cadence"
	
	def check(self, penultimate_chord, ultimate_chord):
		pass