print("hello from analyzeaudio.py")

import tensorflow as tf
from pydub import AudioSegment
import tensorflow_io as tfio

import tensorflow_hub as hub
from concatlabelfiles import *




yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)



my_classes = ['bee', 'combine','goose','human','insect', 'other', 'siren', 'traffic']




def resayhallo():
    print("halloooo")


def loadUp(path):
    return tf.keras.models.load_model(path)


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







def framez(modzel, path, outputFilePath, frameLength, frameHop, concat):

  audio_data = load_wav_16k_mono(path)
  splitted_audio_data = tf.signal.frame(audio_data, frameLength*16, frameHop*16, pad_end=True, pad_value=0)


  with open(outputFilePath,'a') as out:
    out.write('start,end,classification,confidence\n')

  for i, data in enumerate(splitted_audio_data):
    scores, embeddings, spectrogram = yamnet_model(data)
    result = modzel(embeddings).numpy()
    #result = modzel(data).numpy()
    
    class_means = result.mean(axis=0)
    #inferred_class = my_classes[result.mean(axis=0).argmax()]
    predicted_class_index = class_means.argmax()
    inferred_class = my_classes[predicted_class_index]
    confidence_score = class_means[predicted_class_index]

    if (inferred_class == "bee"):
        with open(outputFilePath,'a') as out:
            out.write(f"{(i*frameHop)/1000},{((i*frameHop)+frameLength)/1000},{inferred_class},{confidence_score}\n")
            if(concat == True):
                concatfile(outputFilePath)
            
    elif (confidence_score < 1.5):
        with open(outputFilePath,'a') as out:
            out.write(f"{(i*frameHop)/1000},{((i*frameHop)+frameLength)/1000},{inferred_class},{confidence_score}\n")
            if(concat == True):
                concatfile(outputFilePath)









#framez(loadedmodel, "./trainingdata/misc/chunk20210806_101900_16bit.wav", "./frameoutput.txt", 500, 250)
#loadedmodel = loadUp("./savedmodels/model3")
#framez(loadedmodel, "./loadingdock/orip/College_Mowed_C_ATI_L6,8_1_20220610_092000.wav", "./frameoutput.txt", 500, 250)
