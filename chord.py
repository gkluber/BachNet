import numpy as np
from utils import get_normalized_pitches


class ChordType(Enum):
	#number of members, doubling
	TRIAD_DOUBLED_ROOT = (3, 0)
	TRIAD_DOUBLED_THIRD = (3, 1)
	TRIAD_DOUBLED_FIFTH = (3, 2)
	SEVENTH_CHORD = 4
	SECONDARY_CHORD

class TriadQuality(Enum):
	#first argument = half steps from root to third, second argument = half steps from root to fifth
	MAJOR = (4, 7)
	MINOR = (3, 7)
	DIMINISHED = (3, 6)
	AUGMENTED = (4, 8)
	
	def __init__(self, third, fifth):
		self.third = third
		self.fifth = fifth
	
	@property
	def matches(self, pitches):
		return (pitches[1] - pitches[2] == self.third) and (pitches[0] - pitches[2] == self.fifth)

class SeventhChordQuality(Enum):
	#first argument = half steps from root to third, second argument = half steps from root to fifth, third argument = half steps from root to seventh
	MAJOR = (4, 7, 11)
	DOMINANT = (4, 7, 10)
	MINOR = (3, 7, 10)
	FULLY_DIMINISHED = (3, 6, 10)
	HALF_DIMINISHED = (3, 6, 11)
	
	def __init__(self, third, fifth, seventh):
		self.third = third
		self.fifth = fifth
		self.seventh = seventh
	
	@property
	def matches(self, pitches):
		return (pitches[2] - pitches[3] == self.third) and (pitches[1] - pitches[3] == self.fifth) and (pitches[0] - pitches[3] == self.seventh)