from librosa import load
from librosa import feature
import numpy as np
from librosa import display
from matplotlib import pyplot as pl
from librosa import power_to_db
from librosa import decompose
import image_engine
import cv2

def get_audio(file=""):
    global img
    try:
        data, rate = load(file)
        S = feature.melspectrogram(y=data, sr=rate, n_mels=128, fmax=8000)
       # S = decompose.nn_filter(S,aggregate=np.median,metric='cosine')
        display.specshow(power_to_db(S, ref=np.max), fmax=8000)

        pl.savefig("/tmp/tmpplot.png")
        pl.close()
        img = image_engine.engine64x("/tmp/tmpplot.png")
        #img  = cv2.imread("/tmp/tmpplot.png",0)
        #img = cv2.resize(img,(128,128))





    except:
        print("Error in file ", file)

    return img
