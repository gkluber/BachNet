import numpy as np
from utils import *

def check(last_chord, this_chord, key_sig): #max num is 17
	if check_spelling(this_chord) and check_diatonicity(this_chord, key_sig) and check_repetition(last_chord, this_chord):
		return check_spelling(this_chord) + check_diatonicity(this_chord, key_sig) + check_repetition(last_chord, this_chord) + check_leading_tone_doubling(this_chord, key_sig) + check_melodic_augmented_seconds(last_chord, this_chord) + check_melodic_tritones(last_chord, this_chord) + check_voice_ranges(this_chord) + check_spacing(this_chord) + check_crossover(this_chord) + check_overlap(last_chord, this_chord) + check_fifths(last_chord, this_chord) + check_octaves(last_chord, this_chord) + check_hidden_fifths(last_chord, this_chord) + check_unequal_fifths(last_chord, this_chord) + check_direct_fifths_and_octaves(last_chord, this_chord) + check_leading_tone_resolution(last_chord, this_chord, key_sig)
	else:
		return 0
def check_repetition(last_chord, this_chord):
	return not (last_chord[0]==this_chord[0] and last_chord[1]==this_chord[1] and last_chord[2]==this_chord[2] and last_chord[3]==this_chord[3])

#chord = np.array
def check_spelling(chord) -> bool:		
	#if duplicates exist, then the chord should be a triad
	chord_mod12 = [x%12 for x in chord]
	if len(chord_mod12) != len(set(chord_mod12)):
		#must be triad to be valid
		if not get_root_position_triad(chord).any() == None:
			return True
	else:
		#must be 7th chord
		pass #TODO
	return False
	
#HACK: remove diatonicity check without losing consistency in chord spelling
def check_diatonicity(chord, key_sig) -> bool:
	key_sig_mod12 = np.array([x.midi%12 for x in key_sig.pitches])
	for pitch in chord:
		if pitch%12 not in key_sig_mod12:
			return False
	return True

#key_sig type is key.KeySignature
def check_leading_tone_doubling(chord, key_sig) -> bool:
	leading_tone = (key_sig.tonic.midi - 1) % 12
	occurences = 0
	for pitch in chord:
		if pitch % 12 == leading_tone:
			occurences += 1
	if occurences > 1:
		return False
	return True
	
def check_melodic_augmented_seconds(last_chord, this_chord) -> bool:
	for i in range(len(last_chord)):
		if abs(last_chord[i] - this_chord[i])%12==3:
			return False
	return True
	
def check_melodic_tritones(last_chord, this_chord) -> bool:
	for i in range(len(last_chord)):
		if abs(last_chord[i] - this_chord[i])%12==6:
			return False
	return True

def check_voice_ranges(chord) -> bool:
	if chord[0] > 79 or chord[0] < 60:
		return False
	if chord[1] > 72 or chord[1] < 55:
		return False
	if chord[2] > 67 or chord[2] < 48:
		return False
	if chord[3] > 60 or chord[3] < 43:
		return False
	return True

def check_spacing(chord) -> bool:
	if chord[0] - chord[1] > 12 or chord[1] - chord[2] > 12:
		return False
	return True

def check_crossover(chord) -> bool:
	return chord[0] > chord[1] and chord[1] > chord[2] and chord[2] > chord[3]
	
def check_overlap(last_chord, this_chord) -> bool:
	if this_chord[0] < last_chord[1] or this_chord[0] < last_chord[2] or this_chord[0] < last_chord[3]:
		return False
	if this_chord[1] > last_chord[0] or this_chord[1] < last_chord[2] or this_chord[1] < last_chord[3]:
		return False
	if this_chord[2] > last_chord[0] or this_chord[2] > last_chord[2] or this_chord[2] < last_chord[3]:
		return False
	if this_chord[3] > last_chord[0] or this_chord[3] > last_chord[2] or this_chord[3] > last_chord[3]:
		return False
	return True

def check_fifths(last_chord, this_chord) -> bool:
	for i in range(len(last_chord)):
		for j in range(i+1, len(last_chord)):
			if (last_chord[i] - last_chord[j]) % 12 == 7 and (this_chord[i] - this_chord[j]) % 12 == 7:
				return False
	return True

def check_octaves(last_chord, this_chord) -> bool:
	for i in range(len(last_chord)):
		for j in range(i+1, len(last_chord)):
			if (last_chord[i] - last_chord[j]) % 12 == 12 and (this_chord[i] - this_chord[j]) % 12 == 12:
				return False
	return True

#unacceptable when the bass steps up by one scale step and the soprano steps up to form a perfect interval
def check_hidden_fifths(last_chord, this_chord) -> bool:
	if abs(last_chord[3] - this_chord[3]) in (1,2) and np.sign(last_chord[3] - this_chord[3])==np.sign(last_chord[0] - this_chord[0]) and abs(this_chord[3] - this_chord[0]) % 12 in (0,5,7):
				return False
	return True
	
def check_unequal_fifths(last_chord, this_chord) -> bool:
	return True #todo
	
def check_direct_fifths_and_octaves(last_chord, this_chord) -> bool:
	if abs(last_chord[3] - this_chord[3]) > 2 and abs(last_chord[0] - this_chord[0]) > 2 and np.sign(last_chord[3] - this_chord[3])==np.sign(last_chord[0] - this_chord[0]) and abs(this_chord[3] - this_chord[0]) % 12 in (0,5,7):
				return False
	return True
	
def check_leading_tone_resolution(last_chord, this_chord, key_sig) -> bool:
	result = True
	if last_chord[0] % 12 == (key_sig.tonic.midi - 1)%12:
		if this_chord[0] != key_sig.tonic.midi:
			result = False
	if last_chord[3] % 12 == (key_sig.tonic.midi - 1)%12:
		if this_chord[3] != key_sig.tonic.midi:
			result = False
	
	return result

#TODO ONLY OCCURS IF THE SEVENTH IS MISSING THE THIRD
def check_doubled_seventh(chord, key_sig) -> bool:
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

def check_cadence(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig):
	return check_perfect_authentic_cadence_triad(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig) or check_imperfect_authentic_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) or check_deceptive_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) or check_phyrgian_half_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) or check_half_cadence_triad(this_chord_mod12, key_sig)
	
def check_perfect_authentic_cadence_triad(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig) -> bool:
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

def check_imperfect_authentic_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) -> bool:
	for i in range(4):
		if key_sig.mode=='major':
			if last_chord_mod12[i] not in V_pitches or last_chord_mod12[i] not in viio_pitches or this_chord_mod12[i] not in I_pitches:
				return False
		elif key_sig.mode=='minor':
			if last_chord_mod12[i] not in V_pitches or last_chord_mod12[i] not in viio_pitches or this_chord_mod12[i] not in i_pitches:
				return False
	return True

#V -> vi or VI
def check_deceptive_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) -> bool:
	for i in range(4):
		if key_sig.mode=='major':
			if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in vi_pitches:
				return False
		elif key_sig.mode=='minor':
			if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in VI_pitches:
				return False
	
	return True

#IV or iv -> I
def check_plagal_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) -> bool:
	for i in range(4):
		if key_sig.mode=='major':
			if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in IV_pitches:
				return False
		elif key_sig.mode=='minor':
			if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in iv_pitches:
				return False
	
	return True

#vi6 -> V (only minor keys)
def check_phyrgian_half_cadence_triad(last_chord_mod12, this_chord_mod12, key_sig) -> bool:
	if key_sig.mode is not 'minor':
		return False
	
	for i in range(4):
		if last_chord_mod12[i] not in V_pitches or this_chord_mod12[i] not in iv6_pitches:
			return False
	return True

#any -> V
def check_half_cadence_triad(this_chord_mod12, key_sig) -> bool:
	for i in range(4):
		if this_chord_mod12[i] not in V_pitches:
			return False
	return True

def reward(last_chord, this_chord, last_chord_root_pos, this_chord_root_pos, key_sig, is_last_chord: bool):
	reward = 0
	error = False
	check_num = check(last_chord, this_chord, key_sig)
	reward += check_num
	if check_num < 17:
		error = True
	if reward_circle_progression(last_chord_root_pos, this_chord_root_pos,key_sig):
		reward += 4
	#hack change this so that the cadence goes at the end of each period 
	if is_last_chord and reward_cadence(last_chord, this_chord, key_sig):
		reward += 2
	
	return reward, error

def reward_circle_progression(last_chord_root_pos, this_chord_root_pos, key_sig) -> bool:
	try:
		if this_chord_root_pos.any() == None or last_chord_root_pos.any() == None:
			return False
	except:
		if this_chord_root_pos == None or last_chord_root_pos==None:
			return False
	if last_chord_root_pos[2] - this_chord_root_pos[2] > 0 and (classify_triad(this_chord_root_pos,key_sig) - classify_triad(last_chord_root_pos,key_sig)) % 8 == 4:
		return True
	if last_chord_root_pos[2] - this_chord_root_pos[2] < 0 and (classify_triad(last_chord_root_pos,key_sig) - classify_triad(this_chord_root_pos,key_sig)) % 8 == 5:
		return True
	return False
	
def reward_cadence(last_chord, this_chord, key_sig) -> bool:
	last_chord_mod12 = np.array([x%12 for x in last_chord])
	this_chord_mod12 = np.array([x%12 for x in this_chord])
	return check_cadence(last_chord, this_chord, last_chord_mod12, this_chord_mod12, key_sig)

#negative reward for second inversion chords?
def reward_second_inversion(last_chord, this_chord) -> bool:
	pass #todo