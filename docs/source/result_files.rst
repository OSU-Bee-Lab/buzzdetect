buzzdetect result files
==========================================

Output file structure
-----------------------

The structure of buzzdetect output files is identical to that of the input files,
with the addition of a "_buzzdetect" tag to file names.

For example, these input audio files:

::

    .
    ├── chicory
    │   ├── 1_104
    │   │   └── 250704_0000.mp3
    │   ├── 1_109
    │   │   └── 250704_0000.wav
    │   └── ...
    ├── mustard
    │   ├── 1_103
    │   │   └── 240904_0000.flac
    │   ├── 1_29
    │   │   └── 240904_0000.ogg
    │   └── ...
    └── ...

::

    .
    ├── chicory
    │   ├── 1_104
    │   │   └── 250704_0000_buzzdetect.csv
    │   ├── 1_109
    │   │   └── 250704_0000_buzzdetect.csv
    │   └── ...
    ├── mustard
    │   ├── 1_103
    │   │   └── 240904_0000_buzzdetect.csv
    │   ├── 1_29
    │   │   └── 240904_0000_buzzdetect.csv
    │   └── ...
    └── ...

Note that the file extension is not factored into the output file name.
This causes collisions if input files have the same name with different extensions
(e.g., ``audio.mp3`` and ``audio.wav``). Because this is an edge case, buzzdetect does not have
a mechanism to correct this conflict, though it does check for and drop all conflicting input files.


Result files: _buzzdetect.csv
------------------------------

rows
^^^^^
Every row of a result file corresponds to a single frame.

columns
^^^^^^^^^

* **start:** the timestamp of the start time of the frame, in seconds
* **activation_*:** the neuron activation value for the given frame. An ``activation_`` column will be created for each of the model's neurons that is selected for output.
* **detection_*:** whether or not a detection was called for the event corresponding to the neuron. You will only see this column in the raw output if you ran your analysis in detection mode.

We strongly encouraging using `buzzr <https://github.com/OSU-Bee-Lab/buzzr>`_ to call detections
*separately* from your buzzdetect analysis.
Detection mode was created in response to concerns that the selection of a threshold value introduces subjectivity
in the analysis and hinders repeatability. While detection mode is potentially more convenient insofar as it
eliminates the need to consider a threshold value, we find that the choice of an appropriate threshold
inescapably requires consideration of the signal:noise ratio of the experiment in question.
All analyses are "subjective" insofar as they require decisions; this is true of all research.

Note that there is no "end" column. Because all frames should have the same frame length for a given file,
an end column is redundant. The default location for result files is within the ``output/`` subdirectory
of the corresponding model's directory. The framelength can be found in the model config if needed.

activations
^^^^^^^^^^^^

buzzdetect outputs the raw activation values for each selected neuron.
The values are not softmaxed, are not calibrated to probabilities, are not centered around 0 or in any other way preprocessed.
The distribution of neuron activations varies meaningfully between models;
for one model, an activation above -1.2 might indicate a 95% chance of a true buzz in that frame,
while the same value corresponds to a 20% chance in another model.
With each model, we report our estimated sensitivity, false-positive-rate, and precision across a range of thresholds at which to call buzzes.
Check for these in the model folder.

saving storage
^^^^^^^^^^^^^^^^^^^^^^^
The total volume of results files from large experiments can reach dozens of gigabytes.
One easy way to save space is to output only the results corresponding to neurons of interest (usually just ins_buzz).

For more extreme space saving, see the ``trim_directory()`` function in the companion package `buzzr <https://osu-bee-lab.github.io/buzzr/reference/trim_directory.html>`_.
Trimming result files to only ins_buzz, rounding activations to 1 decimal (from 2), and saving as a compressed RDS
results in a 29:1 compression ratio with no meaninful information loss.
The results also read more quickly than a CSV!
