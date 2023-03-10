import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

main_path = './images/'

styles = pd.read_csv('styles.csv')
categories = styles.masterCategory #uzimanje kolona masterCategory i id
id = styles.id.to_numpy()
print('Prisutne kategorije: ')
print(categories.unique())
no_of_images = []
print('Broj primera svake kategorije: ')
for c in categories.unique(): #nalazenje broja primera za svaku od klasa
    no_of_images.append(np.sum(categories==c))
    print(c + ' - ' + str(np.sum(categories==c)))


classes = categories.unique()[:4] #uzimanje 4 najbrojnije klase
no_of_images = np.array(no_of_images)[:4]

print('Posmatrane kategorije: ')
print(classes)

plt.figure() #histogram
plt.bar(range(len(no_of_images)), no_of_images, align='center')
plt.xticks(range(len(classes)), classes, size='small')
plt.title('Veličine ulaznih skupova za svaku kategoriju')
plt.show()


img_index = np.array(categories.index[categories=='Apparel'][:2])
img_index = np.concatenate((img_index, np.array(categories.index[categories=='Accessories'][:2])))
img_index = np.concatenate((img_index, np.array(categories.index[categories=='Footwear'][:2])))
img_index = np.concatenate((img_index, np.array(categories.index[categories=='Personal Care'][:2])))

N=1
plt.figure() #prikaz slika za svaku od klasa
for i in img_index:
    img = cv2.cvtColor(cv2.imread(main_path + str(id[i]) + '.jpg'), cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,N)
    plt.imshow(img)
    plt.axis('off')
    plt.title(categories.iloc[i])
    N=N+1
plt.show()

resc_dim = (64, 84)

for c, i in zip(categories, id):
    if c in classes:
        try:
            img = cv2.imread(main_path + str(i) + '.jpg')
            img_resc = cv2.resize(img, resc_dim, interpolation=cv2.INTER_CUBIC) #promena dimenzija na 64x84
            img_resc = img_resc[10:74, :] #odsecanje 10 gornjih i 10 donjih piksela, ostaju dimenzije 64x64
            cv2.imwrite('./dataset/' + c + '/' + str(i) + '.jpg', img_resc) #čuvanje slike u odgovarajući folder
        except Exception as e:
            print(str(e) + str(i))
