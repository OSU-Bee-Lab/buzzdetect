# import analyzeaudio

from analyzeaudio import framez
from analyzeaudio import loadUp

#import importlib
#importlib.reload(analyzeaudio)
# loadedmodel = tf.keras.models.load_model("./models/model3")


loadedmodel = loadUp("models/model3")

import os



audiodir = "./loadingdock/togo"
 

for filename in os.listdir(audiodir):
    f = os.path.join(audiodir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        #f = open(f"./bulkoutput/{filename}.txt", "x") 
        framez(loadedmodel, f, f"./bulkoutput/{filename}.txt", 1000, 500)


