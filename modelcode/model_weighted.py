# Imports
#
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from buzzcode.tools import load_wav_16k_mono

# Loading YAMNet from TensorFlow Hub
#
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# model control
modelName = "model_weighted"
epochs_in = 40

model_path = os.path.join("models/", modelName)

if os.path.exists(model_path):
    raise FileExistsError('a model folder with this name already exists; delete or rename the folder and re-run')
else: (os.makedirs(model_path))

# Acquiring and filtering training data
#
metadata_all = pd.read_csv("./training/metadata_all.csv")

classes = [
    # bee lab categories
    "bee",
    "combine",
    "goose",
    "human",
    "insect",
    "other",
    "siren",
    "traffic",
    "train",

    # ESC-50 categories
    "rain",
    "crickets",
    "chirping_birds",
    "wind",
    "airplane"
    #"water drops",
    #"thunderstorm"
]

metadata = metadata_all[metadata_all["category"].isin(classes)]

cat_counts = metadata["category"].value_counts()
cat_weights = (1/cat_counts)*(sum(cat_counts)/len(cat_counts))
map_class_to_weight = {index: weight for index, weight in enumerate(cat_weights) } # I think enumarate counts from 0 when I want to count from 1?

training_data = './training/audio'

class_path = os.path.join(model_path, "classes.txt")

with open(class_path, "w") as file:
    for item in classes:
        file.write(item + "\n")

map_class_to_id = {name: index for index, name in enumerate(classes)}

metadata = metadata.assign(target_model=metadata['category'].apply(lambda name: map_class_to_id[name]))
metadata = metadata.assign(filepath=metadata['filename'].apply(lambda row: os.path.join(training_data, row)))

# Load the audio files and retrieve embeddings
#

filenames = metadata['filepath']
targets = metadata['target_model']
folds = metadata['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

main_ds = main_ds.map(load_wav_for_map)

# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()

# Split the data
#

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

#
# Create your model
#

my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(classes))
], name='my_model')

my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

history = my_model.fit(train_ds,
                       epochs=epochs_in,
                       validation_data=val_ds,
                       callbacks=callback,
                       class_weight=map_class_to_weight)


my_model.save(os.path.join("models", modelName), include_optimizer=True)
