note_names_table = {
	2 : 'D',
	7 : 'G',
	9 : 'A',
}

def get_note(pitch, keysig):
	pitch = pitch % 12

#note that this is immutable
def get_normalized_pitches(chord):
	copy = np.fromiter((lambda x: x % 12 for pitch in chord), chord.dtype, count=len(chord)) #applies mod 12 to all chord members
	copy.sort()
	return copy

def get_root_position(pitches):
	#TODO

def is_first_inversion(chord):
	pitches = get_normalized_pitches(chord)
	