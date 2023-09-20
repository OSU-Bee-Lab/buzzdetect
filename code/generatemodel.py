#
# Import tensorflow and other libraries
#
import os
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

model_name = "fieldClasses_beelab_equalsample"
model_path = os.path.join("models/", model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)


#
# Loading YAMNet from TensorFlow Hub
#
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Utility functions for loading audio files and making sure the sample rate is correct.
@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#
# Load the class mapping
#

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])


#
# Training data
#

training_metadata = './trainingdata/training_metadata_equalsample.csv'
training_data = './trainingdata/audio_equalsample'

pd_data = pd.read_csv(training_metadata)

#
# Filter the data
#

my_classes = pd_data.category.unique()
#my_classes = ['bee', 'combine','goose','human','insect', 'other', 'siren', 'traffic']

class_path = os.path.join(model_path, "classes.txt")

with open(class_path, "w") as file:
    for item in my_classes:
        file.write(item + "\n")

#map_class_to_id = {'bee':0, 'combine':1, 'goose':2,'human':3,'insect':4, 'other':5, 'siren': 6, 'traffic':7}
map_class_to_id = {name: index for index, name in enumerate(my_classes)}

filtered_pd = pd_data[(pd_data.category.isin(my_classes)) & (pd_data.origin == "OSU_beelab")]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(training_data, row))
filtered_pd = filtered_pd.assign(filename=full_path)

#
# Load the audio files and retrieve embeddings
#

filenames = filtered_pd['filename']
targets = filtered_pd['target']
folds = filtered_pd['fold']

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

#
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
    tf.keras.layers.Dense(len(my_classes))
], name='my_model')

#my_model.summary()

my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

history = my_model.fit(train_ds,
                       epochs=20,
                       validation_data=val_ds,
                       callbacks=callback)


my_model.save(os.path.join("models", model_name), include_optimizer=True)

#
# Make a model that operates directly on wav files
#

# class ReduceMeanLayer(tf.keras.layers.Layer):
#   def __init__(self, axis=0, **kwargs):
#     super(ReduceMeanLayer, self).__init__(**kwargs)
#     self.axis = axis
#
#   def call(self, input):
#     return tf.math.reduce_mean(input, axis=self.axis)
#
# saved_model_path = './models/model_luke'
#
# input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
# embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
#                                             trainable=False, name='yamnet')
# _, embeddings_output, _ = embedding_extraction_layer(input_segment)
# serving_outputs = my_model(embeddings_output)
# serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
# serving_model = tf.keras.Model(input_segment, serving_outputs)
#
# serving_model.compile(optimizer='adam', loss='binary_crossentropy')
# serving_model.save(saved_model_path, include_optimizer=True)


