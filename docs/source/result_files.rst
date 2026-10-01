buzzdetect result files
==========================================

Output file organization
--------------------------

The organization of buzzdetect output files is identical to that of the input files,
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


Will output these result files:
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


Idents
-------

Internally (and in `buzzr <https://github.com/OSU-Bee-Lab/buzzr>`_, `SeeNote <https://github.com/lukehearon/seenote>`_, and `FileSync <https://github.com/OSU-Bee-Lab/filesync>`_), we track connect source audio to output results using an "ident".
You can construct the ident as the relative path from the audio directory to the file, dropping the extension.
For result files, we also drop the "_buzzdetect" and "_buzzpart" suffixes.
All of the following paths share an ident:

- ``~/Documents/recordings/Pollinator Survey/Summer 2026/Springfield/1_13/260703_1334.mp3``
- ``C:\Users\Jane Doe\Downloads\buzzdetect analysis FINAL\Pollinator Survey\Summer 2026\Springfield\1_13\260703_1334_buzzpart.csv``
- ``/Users/luke/buzzdetect/results/Pollinator Survey/Summer 2026/Springfield/1_13/260703_1334_buzzdetect.csv``

Assuming the experiment directory name is "Pollinator Survey" and the directory above that is the results or audio directory,
the ident is
``Pollinator Survey/Summer 2026/Springfield/1_13/260703_1334``

Note that the file extension is not factored into the output file name.
This causes collisions if input files have the same name with different extensions.
``audio.mp3`` and ``audio.wav`` in the same folder will have the same ident.
We can't change the result files to write to, e.g., ``audio_mp3_buzzdetect.csv`` because that changes the ident. 
Consequently, buzzdetect checks for and skips over all input files with conflicting idents, leaving it to the user to rename appropriately.

One more thing: the ident method means that *structure is data.* If you reorganize your audio but not the results, you've broken the link between the two.
This will cause buzzdetect to re-analyze the files on the next run and you'll end up with duplicates.
You can use our utility FileSync to manipulate audio files and buzzdetect results in multiple locations at once.

Result files: _buzzdetect.csv
------------------------------

rows
^^^^^

Our models break audio into "frames," discrete chunks of audio that the model attempts to classify.
Every frame gets a score for every class the model has been trained to detect.
For our YAMNet models, the frame length is 0.96s; the model takes in
roughly a second of audio and evaluates it for the presence of
insect buzzing, passing planes, humans talking, the pitter-patter of rain, and then reports those scores to be written to the results file.

Historically, buzzdetect had a variable "frame hop," where you could extract many frames from a small segment of audio (or else very few from a long segment).
We found that this did not improve model performance while adding a lot of complexity within buzzdetect and across our other apps.
We have removed variable frame hop as of buzzdetect v2.0, but vestiges remain in the underlying engine if you want to experiment with it.

columns
^^^^^^^^^

Result files have a variable number of columns with the following names:

* **start:** the timestamp of the start time of the frame, in seconds. Present in every file.
* **activation_*:** the activation value each neuron for the given frame. The model has one neuron for each class it has been trained to detect.
  For example, the insect buzzing neuron ``ins_buzz`` is written to the column ``activation_ins_buzz``.

Prior to version 2.0, buzzdetect offered a detection mode that applied thresholds to the activations *before* they were written to the result file.
We have removed this feature, as the choice of threshold is dependent on model, acoustic environment, and target species.
The appropriate threshold demands consideration of the signal:noise ratio of the experiment in question and so cannot be universal.
However, we document evaluations and recommended starting points in each model's README.
Use our companion R package `buzzr <https://github.com/OSU-Bee-Lab/buzzr>`_ to apply detection thresholds in a reproducible and flexible way.

Note that there is no "end" column. Because all frames have the same frame length for a given output directory, an end column is redundant.

activations
^^^^^^^^^^^^

buzzdetect outputs the raw activation values for each selected neuron.
The values are not softmaxed, are not calibrated to probabilities, are not centered around 0 or in any other way preprocessed.
The distribution of neuron activations varies meaningfully between models;
for one model, an activation above -1.2 might indicate a 95% chance of a true buzz in that frame,
while the same value corresponds to a 20% chance in another model.
For each model, we report our estimated sensitivity, false-positive-rate, and precision at suggested thresholds.
Read them in the model's README under **View Models** in the app (see :doc:`gui`).

saving storage
^^^^^^^^^^^^^^^^^
The total volume of results files from large experiments can reach dozens of gigabytes.
One easy way to save space is to output only the results corresponding to neurons of interest (usually just ins_buzz).

For more extreme space saving, see the ``trim_directory()`` function in the companion package `buzzr <https://osu-bee-lab.github.io/buzzr/reference/trim_directory.html>`_.
Trimming result files to only ins_buzz, rounding activations to 1 decimal (from 2), and saving as a compressed RDS
results in a 29:1 compression ratio with no meaningful information loss.
The results also read more quickly than a CSV!


Result metadata: buzzdetect_manifest.json
--------------------------------------------

Every output directory gets a ``buzzdetect_manifest.json`` file recording the settings that shape its results:

* **modelname:** the model used for analysis.
* **classes_out:** which classes' activations are written.
* **full_quality_decode:** whether audio was resampled with the full-quality filter (see :doc:`gui`).
* **framelength_s:** the length of each frame in seconds, so the end time of a frame is ``start + framelength_s``.
* **thresholds:** the model's suggested detection threshold for each class, when it has them.
  buzzr and SeeNote read these from here.
* **dir_audio** / **dir_out:** the folders used by the most recent run.

The first three are locked: a later analysis into the same folder must use the same model, classes, and decoding,
so every result file in a folder is comparable. When you pick a folder that already has results, the app locks those settings to match.
