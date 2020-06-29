import os
import numpy as np
import convertMelSpec

# change actually path images

path_fotos = "/home/abdux/Downloads/Speech/"

words_cat = os.listdir(path_fotos)
"""file = open("category", "w")
a = 0
for i in words_cat:
    stre = i + "\t" + str(a) + "\n"
    file.writelines(stre)
    a += 1
file.close()"""

audTR = []

audTST = []

print("Datas getting")
COUNTER = 0
for cat in words_cat:
    path_cat = path_fotos + cat
    directory = os.listdir(path_cat)

    directory_size = len(directory)
    TRAIN_LENGHT = directory_size * 0.8
    TEST_LENGHT = directory_size - TRAIN_LENGHT
    COUNTER = 0
    np.random.shuffle(directory)

    catnum = words_cat.index(cat)
    for i in directory:
        path = path_cat + "/" + i

        if path.endswith(".wav"):
            if COUNTER <= TRAIN_LENGHT:
                print("train"+path)

                audio_tr = convertMelSpec.get_audio(file=path)
                audTR.append([audio_tr, catnum])

                COUNTER += 1
            else:
                print("test",path)
                audio_tst = convertMelSpec.get_audio(file=path)
                audTST.append([audio_tst, catnum])



np.save("images_arrays/TRAINAUDfromMelSp", audTR)
np.save("images_arrays/TESTAUDfromMelSp",audTST)
