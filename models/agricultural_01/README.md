README for model: agricultural_01
Date: 2024-02-22

# General
This model is the current best performing buzzdetect model as of 2024-02-22. We named this model "agricultural" because our current training data are all from agricultural environments. The model may be able to generalize to other environments, but at the time of writing, this has not been tested.

# Metadata
This model uses an "intermediate" level of metadata. The mech_auto_* taxa were merged into the single mech_auto and the mech_plane_* categories were merged into mech_plane.

# Performance
This model performs exceptionally well on the test set. The true positive rate for ins_buzz_bee (the honey bee label) is 0.87. False positives for honey bees from abiotic noise are low, with FPR reaching 0.01 for only ambient_day and mech_hum_traffic.

High buzzes yield only a 0.31 true positive rate, a rate less than the rate of false honey bee positives given high buzzes. High buzzes have a low false positive rate from non-buzzes.

Low buzzes yield a slightly better true positive rate at 0.54, but they are often misclassified as non-buzzes. Oddly, trains yield a fairly high rate of false positives for low buzzes.

Within-buzz confusion is moderate to high, showing attraction to the honey bee class (that is, within-buzz error is often a result of calling a honey bee buzz). The cost of this confusion is dependent on application; in our agricultural sites (soybean fields), we find that the overwhleming majority of buzzes are from honey bees, so the frequency of false positives should be quite low. This model would not perform well where the frequency of high buzzes was much higher.

The model shows attraction to ambient_day, ambient_night, mech_auto, and mech_hum_traffic, but most of the resulting confusion is outside of buzzes. Because buzzdetect is intended to be a monitoring tool and not a general audio classifier, we consider outsize-buzz confusion (e.g. calling mech_auto when given mech_plane) to be unconcerning so long as within-buzz confusion is low and most importantly that false positives from non-buzzes are low.

See the confusion table (plotted in confusion.svg, confusion metrics in confusion.csv) for more detail.
