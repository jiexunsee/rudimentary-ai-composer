from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mido

########### PROCESS MIDI FILE #############
mid = MidiFile('allegroconspirito.mid') # a Mozart piece

notes = []

time = float(0)
prev = float(0)

for msg in mid:
	### this time is in seconds, not ticks
	time += msg.time
	if not msg.is_meta:
		### only interested in piano channel
		if msg.channel == 0:
			if msg.type == 'note_on':
				# note in vector form to train on
				note = msg.bytes() 
				# only interested in the note and velocity. note message is in the form of [type, note, velocity]
				note = note[1:3]
				note.append(time-prev)
				prev = time
				notes.append(note)
###########################################

######## SCALE DATA TO BETWEEN 0, 1 #######
t = []
for note in notes:
	note[0] = (note[0]-24)/88
	note[1] = note[1]/127
	t.append(note[2])
max_t = max(t) # scale based on the biggest time of any note
for note in notes:
	note[2] = note[2]/max_t
###########################################

############ CREATE DATA, LABELS ##########
X = []
Y = []
n_prev = 30
# n_prev notes to predict the (n_prev+1)th note
for i in range(len(notes)-n_prev):
	x = notes[i:i+n_prev]
	y = notes[i+n_prev]
	X.append(x)
	Y.append(y)
# save a seed to do prediction later
seed = notes[0:n_prev]
###########################################

############### BUILD MODEL ###############
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(n_prev, 3), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, input_shape=(n_prev, 3), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('linear'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='mse', optimizer='rmsprop')
model.fit(X, Y, batch_size=300, epochs=400, verbose=1)
###########################################

############ MAKE PREDICTIONS #############
prediction = []
x = seed
x = np.expand_dims(x, axis=0)

for i in range(3000):
	preds = model.predict(x)
	print (preds)
	x = np.squeeze(x)
	x = np.concatenate((x, preds))
	x = x[1:]
	x = np.expand_dims(x, axis=0)
	preds = np.squeeze(preds)
	prediction.append(preds)

for pred in prediction:
	pred[0] = int(88*pred[0] + 24)
	pred[1] = int(127*pred[1])
	pred[2] *= max_t
	# to reject values that will be out of range
	if pred[0] < 24:
		pred[0] = 24
	elif pred[0] > 102:
		pred[0] = 102
	if pred[1] < 0:
		pred[1] = 0
	elif pred[1] > 127:
		pred[1] = 127
	if pred[2] < 0:
		pred[2] = 0
###########################################

###### SAVING TRACK FROM BYTES DATA #######
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for note in prediction:
	# 147 means note_on
	note = np.insert(note, 0, 147)
	bytes = note.astype(int)
	print (note)
	msg = Message.from_bytes(bytes[0:3]) 
	time = int(note[3]/0.001025) # to rescale to midi's delta ticks. arbitrary value for now.
	msg.time = time
	track.append(msg)

mid.save('new_song.mid')
###########################################
