import numpy as np
from music21 import key

#note that this is immutable
#UNUSED
def get_normalized_pitches(pitches):
	copy = np.fromiter((lambda x: x % 12 for pitch in pitches), pitches.dtype, count=len(pitches)) #applies mod 12 to all chord members
	return copy

def remove_doubling(pitches):
	for i in range(len(pitches)):
		for j in range(i+1,len(pitches)):
			if pitches[i]%12 == pitches[j]%12:
				return np.delete(pitches, i)
	return pitches

def get_root_position_triad(pitches):
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
	return None
	
def is_root_position_seventh(pitches):
	fifthToSeventh = pitches[0] - pitches[1]
	return is_root_position_triad(pitches[1::]) and (fifthToSeventh in (3,4))
	
def is_root_position_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (3,4)) and (thirdToFifth % 12 in (3,4))

def is_first_inversion_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (3,4)) and (thirdToFifth % 12 in (4,5,6)) and (bassToFifth % 12 in (8,9))

def is_second_inversion_triad(pitches):
	if len(pitches) > 3:
		pitches = remove_doubling(pitches)
	bassToThird = pitches[1] - pitches[2]
	thirdToFifth = pitches[0] - pitches[1]
	bassToFifth = pitches[0] - pitches[2]
	return (bassToThird % 12 in (4,5,6)) and (thirdToFifth % 12 in (3,4)) and (bassToFifth % 12 in (8,9))