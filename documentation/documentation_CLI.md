The main program buzzdetect.py supports three functions:
Train (generates the model)
Analyze (analyzes audio and outputs bee detection times)
Preprocess (Converts 
Each function is described in turn below.


--action train
Generates the model.
--saveto


--action analyze
Analyzes audio and outputs bee detection times.

--type (required)
bulk: expects --src flag to be a directory containing (only) audio files for analysis
single: expects –src flag to be a single audio file

--analyzesrc (required)
Path to audio to analyze. Should be either a directory or a single file (see --type)

--modelsrc (required)
Path to model to use in analyzing audio

--framelength (default 1000)
Length of time, in milliseconds, of each frame of audio to be analyzed

--framehop (default 500)
Length of time, in milliseconds, between the start time of each frame (should be equal to or less than --framelength value

--concat (default false)
false: merge overlapping OR touching audio frames into each other (e.g., audio at [0.0,1), [0.5,1.5), and [1.5-2.5) will be merged into a single entry spanning from 0-2.5)
true: Skip this concatenation process
	

--action preprocess


