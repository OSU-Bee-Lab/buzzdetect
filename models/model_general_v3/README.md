README for model: model_general_v3

Trained: 2025-06-03

## General
This is our best performing general model as of 2025-06-03. The model is "general" in the sense that it classifies all insect flight buzzes under the "ins_buzz" label. It makes no attempt to further identify the insects. It also conflates various other labels; see translation.csv for details.

This model is useful for application in a study system dominated by a single species (e.g., in our studies in soybean, we find that the majority of buzzes are from honey bees) or for tagging buzzes for further manual analysis. In the future, we intend to use a general model as a gate to expert submodels, using a mixture-of-experts approach for fine-grained identification of pollinators.

Performance is still uneven on different pitches of buzzes (see ./tests/metrics_buzz.svg), with greater sensitivity to medium (near honey bee pitch) buzzes and essentially no sensitivity to high pitched buzzes. Note that the test set itself did not contain many high pitched buzzes, so the 0 sensitivity value is a coarse estimate.

## Training
All of the training audio for model_general was produced in experiments conducted in soybean fields during bloom in central Ohio. The majority of buzzes in the training set are from honey bees (as identified by comparison to reference audio).

Training samples were taken with a framehop of 0.096s (0.10*framelength) and a minimum event overlap of 0.192s (0.20*framelength). That is, the first frame would be placed such that the event started 0.192s from the end of the frame. The second frame would be placed 0.096s after the first, and so forth until a frame contained less than 0.192s of the event.

See the file weights.csv in this the model for the training volume distribution across training set classes.


## Performance
See the plots in the "tests" directory for a detailed exploration of model performance. We recomment a default theshold of -1.1 for 95% precision and 27% sensitivity.
