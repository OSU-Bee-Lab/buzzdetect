README for model: model_general

Date: 2025-02-05 (trained 2024-04-25)

## General
This is our best performing general model as of 2025-02-05 \[Update: now succeeded by model_general_v3! At least, for sensitivities above 95%\]. The model is "general" in the sense that it classifies all insect flight buzzes under the "ins_buzz" label. It makes no attempt to further identify the insects.

This model is useful for application in a study system dominated by a single species (e.g., in our studies in soybean, we find that the majority of buzzes are from honey bees) or for tagging buzzes for further manual analysis. In the future, we intend to use a general model as a gate to expert submodels, using a mixture-of-experts approach for fine-grained identification of pollinators.

## Training
All of the training audio for model_general was produced in experiments conducted in soybean fields during bloom in central Ohio. The majority of buzzes in the training set are from honey bees (as identified by comparison to reference audio).

Training samples were taken with a framehop of 0.096s (0.10*framelength) and a minimum event overlap of 0.192s (0.20*framelength). That is, the first frame would be placed such that the event started 0.192s from the end of the frame. The second frame would be placed 0.096s after the first, and so forth until a frame contained less than 0.192s of the event. This yielded 27,717 training samples of buzzing audio.

See the file weights.csv in this the model for the training volume distribution across training set classes.


## Performance
For buzzes, a threshold of -1.45 yields roughly 90% specificity and 42% sensitivity.

See the model directory for graphs summarizing performance \[though, at the time of writing, these graphs are under reconstruction. If they aren't present...email the authors!\]
