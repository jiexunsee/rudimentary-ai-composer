# Rudimentary AI Music Composer
The aim was to do a very basic AI music composer in 100 lines of code. This script takes a midi file and extracts just the notes, converting the piece into a large array of notes.

It uses the mido library for the handling of midi files and messages. The music I used was a Allegro con spirito by Mozart from [Classical Piano Midi Page](http://www.piano-midi.de/mozart.htm)

The model is built on Keras, using the LSTM layer. Since each note is a vector of pitch, velocity, and time, I trained the model to predict the next note given an array of n_prev previous notes (where n_prev is a parameter to be tuned).

Currently, it does not produce any intelligible music, but hopefully with some parameter tuning it could work a little better. It would probably help a lot to give it some help with musical theoretic features and syntax instead of letting it learn from just raw notes and timings. To do...
