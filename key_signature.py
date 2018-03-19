import numpy as np
from enum import Enum

class KeySignature(Enum):
	C_MAJOR = (0,True)
	G_MAJOR = (1,True)
	D_MAJOR = (2,True)
	A_MAJOR = (3,True)
	E_MAJOR = (4,True)
	B_MAJOR = (5,True)
	F_SHARP_MAJOR = (6,True)
	C_SHARP_MAJOR = (7,True)
	G_SHARP_MAJOR = (8,True)
	D_SHARP_MAJOR = (9,True)
	A_SHARP_MAJOR = (10,True)
	F_MAJOR = (-1,True)
	B_FLAT_MAJOR = (-2,True)
	E_FLAT_MAJOR = (-3,True)
	A_FLAT_MAJOR = (-4,True)
	D_FLAT_MAJOR = (-5,True)
	G_FLAT_MAJOR = (-6,True)
	C_FLAT_MAJOR = (-7,True)
	
	#minor keys. let num_sharps be relative to c-minor
	C_MINOR = (0,False)
	G_MINOR = (1,False)
	D_MINOR = (2,False)
	A_MINOR = (3,False)
	E_MINOR = (4,False)
	B_MINOR = (5,False)
	F_SHARP_MINOR = (6,False)
	C_SHARP_MINOR = (7,False)
	G_SHARP_MINOR = (8,False)
	D_SHARP_MINOR = (9,False)
	A_SHARP_MINOR = (10,False)
	F_MINOR = (-1,False)
	B_FLAT_MINOR = (-2,False)
	E_FLAT_MINOR = (-3,False)
	A_FLAT_MINOR = (-4,False)
	D_FLAT_MINOR = (-5,False)
	G_FLAT_MINOR = (-6,False)
	C_FLAT_MINOR = (-7,False)
	
	def __init__(self, num_sharps, major):
		
		offset = num_sharps * 7 #skips by fifths (interval = 7 half steps) from C4
		
		self.tonic = offset % 12 + 60 #starts at C4 = 60
		
		self.accs = [] #accidentals
		accidental_offset = 0 if major else -21
		
		for _ in range(abs(num_sharps) if major else abs(num_sharps - 3)):
			accidental_offset += 7
			note = (accidental_offset - 2 + self.tonic) % 12 + 60
			self.accs.append(note)
		
		self.accs.sort()
		
		self.major = major
		self.sharp = num_sharps >= 0 if major else num_sharps - 3 >= 0 #determines the identity of accidentals
		
		self.build_scale()
		
	def build_scale(self):
		formula = [2,2,1,2,2,2,1] if self.major else [2,1,2,2,1,2,2] #major and minor scale formulas
		
		self.pitches = []
		self.pitches.append(self.tonic)
		
		accumulatedInterval = 0
		
		for interval in formula:
			accumulatedInterval += interval
			self.pitches.append(self.tonic + accumulatedInterval)
	
	@property
	def tonic(self);
		return self.tonic
	
	@property
	def scale(self):
		return self.pitches
	
	#TODO: below functions only work when abs(num_sharps) < 8
	@property
	def accidentals(self):
		return self.accs
	
	#TODO
	def __str__(self):
		notes = []
		'''if self.sharp:
			for pitch in self.pitches:
				if pitch in KeySignature.C_MAJOR.scale:
					notes.append
		'''
				
	
	