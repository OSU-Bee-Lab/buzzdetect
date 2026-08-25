## At a glance
### Purpose
This model detects the buzzing of insect flight, outputting the activation to the "activation_ins_buzz" column of the results.

### Recommended threshold
In our tests, a threshold of -1.2 corresponded to a precision of 95%, a sensitivity of 28%, and a false positive rate of ~0.3%. We recommend this as a starting point for calling detections. Lower thresholds quickly become prone to false positives. Higher thresholds may still yield strong results in recordings with many buzzes.

See [metrics.svg](tests/metrics.svg) for a graphical view of the trade-offs of different threshold levels.

### Shortcomings
**Nighttime false positives.** We've found that there are often small occurrences of false positives from cricket calls at night. On average across a large swathe of our dataset, we see nighttime (from 9PM to 6AM in summer in Columbus, OH) detection rates around 0.07% (~3 per hour). In comparison, average daytime detection rates are 3% overall, 1%-8% in high-activity environments (e.g., fields of mustard cover crop in bloom). Still, these false positives can sometimes register on graphs of daily patterns, especially in low-activity environments

**Unequal sensitivity.**  This model is insensitive to high-pitched buzzes, as noted in [our paper](https://doi.org/10.1093/jisesa/ieaf104). See [metrics_buzz.svg](tests/metrics_buzz.svg) for a comparison of the differential sensitivity to different pitches of buzzes at a given level of precision.

Early tests in new environments indicate that the model could be insensitive to what we would consider medium-pitched buzzes of native bees, namely _Mellisodes spp._ Researchers should validate results from new environments or new target species. Efforts to expand the diversity of our training dataset are ongoing. Want to try buzzdetect in your research? We'd be happy to collaborate!

## General
This is our best performing general model as of 2025-06-03. The model is "general" in the sense that it classifies all insect flight buzzes under the "ins_buzz" label. It makes no attempt to further identify the insects. See [the translation file](translation.csv) for more information about how labels were simplified for training.

This model is useful for application in a study system dominated by a single species (e.g., in our studies in soybean, we find that the majority of buzzes are from honey bees) or for tagging buzzes for further manual analysis. In the future, we intend to use a general model as a gate to expert submodels, using a mixture-of-experts approach for fine-grained identification of pollinators.


## Training
All of the training audio for model_general was produced in experiments conducted in soybean fields during bloom in central Ohio. The majority of buzzes in the training set are from honey bees (as identified by comparison to reference audio).

Training samples were taken with a framehop of 0.096s (0.10\*framelength) and a minimum event overlap of 0.192s (0.20\*framelength). That is, the first frame would be placed such that the event started 0.192s from the end of the frame. The second frame would be placed 0.096s after the first, and so forth until a frame contained less than 0.192s of the event.

See the file weights.csv in this the model for the training volume distribution across training set classes.
