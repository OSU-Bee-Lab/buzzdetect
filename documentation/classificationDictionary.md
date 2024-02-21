# Buzzdetect Classification Dictionary
Version 2.1 (in progress)

### Version 2.1 changelist
* Added 
   * ambient_bang
   * ambient_rustle
* Renamed
   * mech_hum to mech_hum_traffic to make room for future non-traffic hums.

## Overview
This dictionary documents all classes of interest for the buzzdetect training set. The dictionary headers are given in the label format with semantic names in parenthesis. E.g., the insect rank is given as “ins (insects)”. Terminal ranks are given with their full label identity. E.g., the honey bee rank has the epithet “bee”, but is listed with the full label: “ins_buzz_bee (honey bee)”

### Specificity
We have chosen to err on the side of specificity in our classifications. We do this for two reasons.
Firstly, we find that our models perform better (that is, more sensitive and more specific buzz identification) with more classes, despite smalling training volume per-class.
Second, it's trivial to merge classifications in training metadata after the fact, but it's impossible to split a classification without manual relabeling of files.

### Hierarchy
At this time, buzzdetect does not use a hierarchical classification model. Nonetheless, the class dictionary is grouped in hierarchies to structure nomenclature, ease analysis of results, and to make room for future updating to a hierarchical model.
The hierarchies are reflected below with nested headings. Terminal ranks—that is, the label applied to the training data—are tagged with `LABEL: `.

Labels are made by concatenating ranks with underscores. For example, the taxonomy of a propeller plane is: mechanical/plane/propeller and would be labeled as mech_plane_prop.

### "RECLASSIFY"
When the class dictionary undergoes changes, it can break previous labels.
For example, when `mech_auto` was split into `mech_auto_truck` and `mech_auto_car`, all previous classifications were left ambiguous.
Ideally, all of these files would be manually reviewed and reclassified. Pragmatically, this is often too large a task to undertake immediately.
For the intervening time, the ambiguous labels are renamed as specifically as possible and tagged with "_RECLASSIFY". Thus, `mech_auto` becomes `mech_auto_RECLASSIFY`.
This allows us to train future models flexibly. The metadata for a model could drop everything tagged with "_RECLASSIFY", use these labels as-is or merge all `mech_auto_*` labels back into `mech_auto`

Some labels in the dataset are simply `RECLASSIFY`. These come from previous labels that were too general to be useful. For example, early iterations of the training set used classifications of `other` and `misc`.
These files were found to represent any number of sounds from `ambient_day` to `ambient_scraping` to `human`.

## ins (insect)
The taxon containing all insect-produced sounds: flight buzzes, chirps, timballing, etc. While we hope to expand these categories in the future, this version of the class dictionary is focused on the detection of honey bees, so categories are only as specific as we feel necessary to distinguish honey bees from other insects.

### buzz
The buzzing hum of insect flight. Most detections appear to be fly-bys, which produce an audible doppler shift. Flower visitations have a two-buzz “visit…leave” pattern. Visitations are not labeled differently than fly-bys, as the frame length of buzzdetect is unlikely to capture this pattern. Some insects (particularly low ones, perhaps bumble bees) hover quite close to the microphone for seconds on end, sometimes producing a rasp of wings on the microphone surface.
These categories are, as of this version, entirely relative to honey bee flight. Uncertain classifiers should find reference audio or seek the ear of an expert!

#### LABEL: ins_buzz_bee (honey bee)
Honey bee (Apis mellifera) flight. 
Classifiers should take note of the pitch of the buzz to distinguish honey bees from other insects. 

#### LABEL: ins_buzz_high (high-pitched buzz)
Flight buzzes that are higher in pitch than that of the honey bee. Perhaps small solitary bees, mosquitoes, flies.

#### LABEL: ins_buzz_low (low-pitched buzz)
Flight buzzes that are lower in pitch than that of the honey bee. Perhaps bumble bees.

### trill
#### LABEL: ins_trill
Sharp, nearby chirping or tymbaling as of crickets, cicadas, katydids.
Note: a distant background of crickets or cicadas should be classified as ambient.

## goose
#### LABEL: goose
goose


## human
#### LABEL: human
Human vocalizations. Not non-vocal human activity (e.g. striding through crops).


## mech (mechanical)
The taxon containing all noises produced by machinery.

### plane
#### LABEL: mech_plane_prop
The hum of a propeller plane in flight.

#### LABEL: mech_plane_jet
The whoosh and scream of a jet plane in flight.

### auto
The taxon for all ground-based vehicles.

#### LABEL: mech_auto_truck
The roar of a large truck passing nearby, distant sounds of downshifting, the low rumble of a diesel engine.

#### LABEL: mech_auto_car
The woosh of a passing car, an engine accelerating onto the main road, street racers at night.

#### LABEL: mech_auto_bike (motorcycle)
The rumble of a chopper, the whine of a sports bike revving.

### combine
#### LABEL: mech_combine
The persistent, mechanical drone of a combine harvester (or other heavy farm equipment) in operation. Consistent sounds of the engine, intermittent sounds of the equipment. May vary in pitch or modulate with the activation of machinery.


### train
#### LABEL: mech_train
The whistle of a train or rumble of a train crossing tracks


### siren
#### LABEL: mech_siren
The siren of an emergency vehicle

### hum
Far off, droning, mechanical noises.

#### LABEL: mech_hum_traffic
The sound of distant (usually highway) traffic. Not the sound of engines or rushing air, but a far off, continuous, fairly high-pitched and slightly wavering drone. Perhaps sound produced by tires on a concrete road.


## ambient
Routine environmental sound not related to particular events of interest.

Note: the day and night sub-ranks of ambient were determined by the clock, not by ear. Current ambient sounds were recorded in July in Ohio. `ambient_day` was considered to be all ambient noise between 6:00am and 9:00pm, corresponding with sunrise and sunset. ambient_night was considered to be all ambient noise between 9:00pm and 6:00am.
Future recordings in a different sonic landscape (different region of the world, different time of year) may need additional training audio or may need to completely refresh the ambient training audio.

### day
#### LABEL: ambient_day
The regular background accompaniment of a fine summer’s day. Sitting out in the middle of a field of soybeans reading a good book.
May contain:
* Birdsong
* The rustling of leaves
Initially, these sounds were chosen programmatically as “three seconds of sound during the day (as defined in the description of ambient) where there is no manual classification for 15 minutes before or after.” Afterwards, all programmatically chosen files were reviewed and manually reclassified if they contained non-ambient sounds.

### night
#### LABEL: ambient_night
The regular background accompaniment of a quiet summer’s night. Laying on a blanket under the stars.
May contain:
* Crickets chirping in the distance (near chirps should be labeled as ins_trill)
* Frogs peeping in the distance
These labels were made by programmatically snipping the first 3s of every night hour (starting at 9:00pm, ending at 6:00am) in every training raw[c] available.

### scraping
#### LABEL: ambient_scraping
The artificially-amplified sound of something scraping over the recorder or its stake. May be caused by any number of sources, from an animal passing by to strong winds. Distinct from the normal, gentle rustle of leaves that should be categorized as ambient.

### rain
#### LABEL: ambient_rain
The pitter patter of raindrops falling.

### bang
Brief, loud, atonal sounds.

#### LABEL: ambient_bang
A gunshot, a car backfiring, a firework.

### rustle
#### LABEL: ambient_rustle
Sounds of wind swishing leaves; hard wind blowing; fabric swishing; not sharp and amplified like ambient_scraping
