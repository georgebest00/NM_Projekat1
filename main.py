import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

main_path = './dataset/'

image_size = (64, 64)
batch_size = 64

from keras.utils import image_dataset_from_directory

X_training = image_dataset_from_directory(main_path,
                                            subset='training',
                                            validation_split=0.3,
                                            image_size= image_size,
                                            batch_size=batch_size,
                                            seed = 123) #obucavajuci skup



classes = np.array(X_training.class_names) #prisutne klase

X_validation = image_dataset_from_directory(main_path,
                                        subset='validation',
                                        validation_split=0.3,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        seed=123)

val_batches = tf.data.experimental.cardinality(X_validation)
X_test = X_validation.take(val_batches // 2) #test skup, bira se pola batch-eva od onih koji nisu u trening skupu
X_validation = X_validation.skip(val_batches // 2) #validacioni skup

from keras import layers
from keras import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

data_augmentation = Sequential([
    layers.RandomFlip('horizontal', input_shape=(image_size[0], image_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1)
]) #slojevi koji vrse prostorne transformacije na slikama

class_count = len(classes)

def make_model(hp): #funkcija koja vraca model sa hiperparametrima izabranim pomocu RandomSearch

    no_of_conv_layers = hp.Int('conv_layers', min_value=3, max_value=5)
    no_of_dense_layers = hp.Int('dense_layers', min_value=1, max_value=3)
    dense_size = hp.Choice('dense_size', values=[64, 128, 256])
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255,input_shape=(64,64,3)), # preskalira slike

        layers.Conv2D(16,3,padding='same', activation='relu'), #mreza da ima bar jedan konvolucioni sloj
        layers.MaxPooling2D(2,2)
    ])

    for i in range(no_of_conv_layers-1): #dodaju se ostali konvolucioni slojevi
        model.add(layers.Conv2D(2**(i+5),3,padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Dropout(0.2)) #dropout sloj za sprecavanje preobucavanja
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu')) #mreza da ima bar jedan Dense sloj

    for i in range(no_of_dense_layers-1): #dodaju se ostali Dense slojevi
        model.add(layers.Dense(dense_size, activation='relu'))

    model.add(layers.Dense(class_count,activation='softmax')) #poslednji sloj

    model.compile(Adam(learning_rate=0.001),
                  loss = SparseCategoricalCrossentropy(),
                  metrics='accuracy') #zadavanje optimizacionog postupka i krit. funkcije

    return model

import keras_tuner as kt

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3) #rano zaustavljanje za sprecavanje preobucavanja

from sklearn.utils import class_weight

y = classes[np.concatenate([y for x, y in X_training], axis=0)]

weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y) #tezinske klase jer su klase
print('Težine klasa: ' + str(weights))                                                     #nebalansirane

tuner = kt.RandomSearch(make_model, objective='val_accuracy', overwrite=True,
                        max_trials=10)

tuner.search(X_training, epochs=20, validation_data=X_validation,
             callbacks = [es],
             class_weight={0:weights[0], 1:weights[1],
                            2:weights[2], 3:weights[3]}) #trazenje najboljih hiperparametara

best_hp = tuner.get_best_hyperparameters()[0]
best_no_of_conv_layers = best_hp['conv_layers']
best_no_of_dense_layers=best_hp['dense_layers']
best_dense_size=best_hp['dense_size']

print("Optimalni broj konvolucionih slojeva: " + str(best_no_of_conv_layers))
print("Optimalni broj potpuno povezanih slojeva: " + str(best_no_of_dense_layers))
print("Optimalni broj neurona u potpuno povezanim slojevima: " + str(best_dense_size))

model = tuner.hypermodel.build(best_hp) #najbolji model

model.summary()

history = model.fit(X_training,
          validation_data=X_validation,
          epochs=20, callbacks=[es], class_weight={0: weights[0], 1: weights[1],
                                    2: weights[2], 3: weights[3]}) #ponovno obucavanje najboljeg modela


i=0
plt.figure() #iscrtavanje lose klasifikovanih primera na test skupu
for img, lab in X_test:
    pred = np.argmax(model.predict(img, verbose=0), axis=1)
    lab = lab.numpy()
    for j in range(batch_size):
        if lab[j] == i and pred[j] != lab[j]:
            plt.subplot(2, 2, i+1)
            plt.imshow(img[j].numpy().astype('uint8'))
            plt.title('Tačna klasa: ' + classes[i] + ', Dobijena klasa: ' + classes[pred[j]])
            plt.axis('off')
            i = i+1
        if i == 4:
            break
    if i == 4:
        break

i = 0
plt.figure() #iscrtavanje dobro klasifikovanih primera na test skupu
for img, lab in X_test:
    pred = np.argmax(model.predict(img, verbose=0), axis=1)
    lab = lab.numpy()
    for j in range(batch_size):
        if lab[j] == i and pred[j] == lab[j]:
            plt.subplot(2, 2, i+1)
            plt.imshow(img[j].numpy().astype('uint8'))
            plt.title('Tačna klasa: ' + classes[i] + ', Dobijena klasa: ' + classes[pred[j]])
            plt.axis('off')
            i = i+1
        if i == 4:
            break
    if i == 4:
        break

plt.figure() #iscrtavanje performansi obučavanja kroz epohe
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Kriterijumska funkcija')
plt.legend(['Trening skup', 'Val skup'])

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Tačnost')
plt.legend(['Trening skup', 'Val skup'])

pred_training = np.array([])
labels_training = np.array([])

for img, lab in X_training: #provera uspešnosti mreže na trening skupu
    labels_training = np.append(labels_training, lab)
    pred_training = np.append(pred_training, np.argmax(model.predict(img, verbose=0), axis=1))

from  sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
print('Tačnost modela na trening skupu je: ' + str(100*accuracy_score(labels_training, pred_training)) + '%')

cm_training = confusion_matrix(labels_training, pred_training, normalize='true') #konfuziona matrica na trening skupu
cm_training_display = ConfusionMatrixDisplay(confusion_matrix=cm_training, display_labels=classes)
cm_training_display.plot()

TP = cm_training[3, 3]
TN = np.sum(cm_training[0:3, 0:3])
FN = np.sum(cm_training[3, 0:3])
FP = np.sum(cm_training[0:3, 3])

print('Preciznost na trening skupu u odnosu na klasu Personal Care: ' + str(TP/(TP+FP)))
print('Osetljivost na trening skupu u odnosu na klasu Personal Care: ' + str(TP/(TP+FN)))

pred_test = np.array([])
labels_test = np.array([])

for img, lab in X_test: #provera uspešnosti mreže na trening skupu
    labels_test = np.append(labels_test, lab)
    pred_test = np.append(pred_test, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tačnost modela na test skupu je: ' + str(100*accuracy_score(labels_test, pred_test)) + '%')

cm_test = confusion_matrix(labels_test, pred_test, normalize='true') #konfuziona matrica na test skupu
cm_test_display = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
cm_test_display.plot()
plt.show()

TP = cm_test[3, 3]
TN = np.sum(cm_test[0:3, 0:3])
FN = np.sum(cm_test[3, 0:3])
FP = np.sum(cm_test[0:3, 3])

print('Preciznost na test skupu u odnosu na klasu Personal Care: ' + str(TP/(TP+FP)))
print('Osetljivost na test skupu u odnosu na klasu Personal Care: ' + str(TP/(TP+FN)))
