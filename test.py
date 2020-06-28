import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import re
import convertMelSpec

from keras import models


fs = 44100
sec = 1
sd.default.device = [0,0]  #MY COMPUTER MIC RUNNING PULSEAUDIO DRIVER

record = sd.rec(int(sec*fs),samplerate=fs,channels=2)
print("recording")
sd.wait()

write("record.wav",fs,record)
print("recorded")
#samplerate, data = wavfile.read("record.wav")


test_aud_fre = convertMelSpec.get_audio("record.wav")


category = []

c = 0

with open("category","r") as file:
    for i in file:
        i = re.sub('\s+', '', i)
        i = i.replace(str(c),"")
        category.append([i,c])
        c+=1



test_aud_fre = test_aud_fre.reshape((1, 64 , 64,1))
test_aud_fre = test_aud_fre.astype('float32') / 255


model =models.load_model("LR-0.0001-ACC-87.10451126098633-batch-128-regu-0.001INP-64_MODEL")

prediction=model.predict([test_aud_fre])

control = np.argmax(prediction)

print("prediction = ", prediction)

for i in category:
    if i[1] == control:
        print("words is :",i[0])


