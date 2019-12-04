# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:02:55 2019

@author: emilio
"""
from __future__ import absolute_import
from __future__ import print_function
print('STEP1: libraries') 
import numpy as np
import keras
#import random
#from keras.datasets import mnist
from keras.models import Model
from keras import models
from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras import backend as K
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import math

K.clear_session()




############################################################### FUNCTIONS #####
print('STEP2: functions')
def load_imgs(img_filenames, image_dir):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    imgs = np.empty((img_filenames.shape[0], 224, 224, 3))
    filenames = img_filenames[['x']]
    
    for i in range(img_filenames.shape[0]):
        img = load_img(os.getcwd() + image_dir + '/' + filenames.iloc[i, 0], target_size=(224, 224))
        imgs[i] = (img_to_array(img) * -255).astype(np.uint8)/255
    return imgs

def load_pairs(panda_df_pairs, imgs_filenames, imgs_processed):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    left_imgs = np.empty((panda_df_pairs.shape[0], 224, 224, 3))
    right_imgs = np.empty((panda_df_pairs.shape[0], 224, 224, 3))
    
    left_filenames = panda_df_pairs[['V1']]
    right_filenames = panda_df_pairs[['V2']]
    
    for i in range(panda_df_pairs.shape[0]):
        position_imgl = img_filenames[['x']].values == left_filenames.iloc[i, 0]
        left_imgs[i] = imgs_processed[np.where(position_imgl == True)[0][0]]
        position_imgr = img_filenames[['x']].values == right_filenames.iloc[i, 0]
        right_imgs[i] = imgs_processed[np.where(position_imgr == True)[0][0]]
    return left_imgs, right_imgs

def euclidean_distance(vects): 
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred): # y = 1 means similar
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.2
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def contrastive_loss_2lims(y_true, y_pred): # y = 1 means similar
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin_inf = 0.1
    margin_sup = 0.2
    margin_inf_loss = K.square(K.maximum(y_pred - margin_inf, 0))
    margin_sup_loss = K.square(K.maximum(margin_sup - y_pred, 0))
    return (0.5 * y_true * margin_inf_loss) + (0.5 * (1 - y_true) * margin_sup_loss)

def create_base_network_vgg16_top_trainable():
    '''Base network to be shared (eq. to feature extraction).
    '''
    vgg16_model = keras.applications.vgg16.VGG16(weights = 'imagenet')
    input_layer = x = vgg16_model.input
    for i, layer in enumerate(vgg16_model.layers[1:22], 1):
        x = layer(x)
    cnnModel = keras.models.Model(inputs = input_layer, outputs = x)
    for layer in cnnModel.layers[0:21]:
        layer.trainable = False  
    del vgg16_model
    return cnnModel


def create_base_network_scratch(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = models.Sequential()
    model.add(keras.layers.Conv2D(64, (3,3), input_shape = input_shape, activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3,3), input_shape = input_shape, activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3,3), input_shape = input_shape, activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(512, (3,3), input_shape = input_shape, activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(512, (3,3), input_shape = input_shape, activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(1024, activation = 'relu'))
    model.add(keras.layers.Dense(1024, activation = 'relu'))
    return (model)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 11, y_true.dtype)))

def generator(samples, imgs_processed, img_filenames, batch_size=32):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = samples.shape[0]
    while True: # Loop forever so the generator never terminates
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples.iloc[offset:offset+batch_size, ]
            # Initialise X_train and y_train arrays for this batch
            Xleft_train = []
            Xright_train = []
            y_train = []
            # For each example
            for i in range(batch_samples.shape[0]):
                # Load image (X)
                position_imgl = img_filenames[['x']].values == batch_samples.iloc[i, 1]
                imgl = imgs_processed[np.where(position_imgl == True)[0][0]]
                #imgl = load_img(os.getcwd() + image_dir + '/' + batch_samples.iloc[i, 1], target_size=(224, 224))
                #imgl = (img_to_array(imgl) * -255).astype(np.uint8)/255
                position_imgr = img_filenames[['x']].values == batch_samples.iloc[i, 2]
                imgr = imgs_processed[np.where(position_imgr == True)[0][0]]
                #imgr = load_img(os.getcwd() + image_dir + '/' + batch_samples.iloc[i, 2], target_size=(224, 224))
                #imgr = (img_to_array(imgl) * -255).astype(np.uint8)/255
                # Read label (y)
                y = batch_samples.iloc[i, 4]
                # Add example to arrays
                Xleft_train.append(imgl)
                Xright_train.append(imgr)
                y_train.append(y)
            # Make sure they're numpy arrays (as opposed to lists)
            Xleft_train = np.array(Xleft_train)
            Xright_train = np.array(Xright_train)
            y_train = np.array(y_train)
            # The generator-y part: yield the next training batch            
            yield [Xleft_train, Xright_train], y_train

###############################################################################



################################################################# DATA AND MODEL
print('STEP3: read pointers and images')
#dir definition
os.chdir('/home/emilio/image-similarity/fashion-dataset/thesis')
#os.chdir('C:\\Users\\emilio\\Documents\\fashion-dataset\\thesis')
image_dir = "/images"
#image_dir = "\\images"

#read pairs indiators
df_pairs_train = pd.read_csv(os.getcwd() + '/training_pairs.csv')
#df_pairs_train = pd.read_csv(os.getcwd() + '\\training_pairs.csv')
df_pairs_val = pd.read_csv(os.getcwd() + '/validation_pairs.csv')
#df_pairs_val = pd.read_csv(os.getcwd() + '\\validation_pairs.csv')
img_filenames = pd.read_csv(os.getcwd() + '/img_filenames.csv')
#img_filenames = pd.read_csv(os.getcwd() + '\\img_filenames.csv')


imgs_processed = load_imgs(img_filenames, image_dir)

train_generator = generator(df_pairs_train, imgs_processed, img_filenames, batch_size=32)
validation_generator = generator(df_pairs_val, imgs_processed, img_filenames, batch_size=32)


 

# network definition
print('STEP4: network creation')
input_shape = (224, 224, 3)

base_network = create_base_network_scratch(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

run = 'scratchmodel_euclidean_cont1lim'
###############################################################################





print('STEP5: load pairs and compute distance before training')
max_batch = 2500
iterations = math.ceil(df_pairs_val.shape[0]/max_batch) 
distances_untr = np.empty((df_pairs_val.shape[0], 1))



if(iterations == 1): #case 1: of pairs_df less than 2500 rows
    left_imgs, right_imgs = load_pairs(df_pairs_val, img_filenames, imgs_processed)
    distances_untr = model.predict([left_imgs, right_imgs])
else: #case 2: of pairs_df more or equal to 2500 rows
    for i in range(iterations):
        print('STEP5.1: compute distance | iteration ' + str(i))
        if(i == 0):
            pairs_sub_df = df_pairs_val.iloc[0:max_batch, ]
            client_imgs, competitor_imgs = load_pairs(pairs_sub_df, img_filenames, imgs_processed)
            distances_untr_batch = model.predict([client_imgs, competitor_imgs])
            distances_untr[0:max_batch] = distances_untr_batch
        else: 
            pairs_sub_df = df_pairs_val.iloc[(i * max_batch):((i + 1) * max_batch), ]    
            client_imgs, competitor_imgs = load_pairs(pairs_sub_df, img_filenames, imgs_processed)
            distances_untr_batch = model.predict([client_imgs, competitor_imgs])
            distances_untr[(i * max_batch):((i + 1) * max_batch)] = distances_untr_batch

 
    
    
    
    
#training
print('STEP6: learning')
epochs = 20


momentum = 0.6
seed = 1124
np.random.seed(seed)

adam = Adam(lr = 0.00002)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

model.compile(loss=contrastive_loss, optimizer=adam, metrics=[contrastive_loss])

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator, validation_steps = 228,
                              steps_per_epoch=2093, nb_epoch = epochs, callbacks = [es])






print('STEP7: load pairs and compute distance after training')
max_batch = 2500
iterations = math.ceil(df_pairs_val.shape[0]/max_batch) 
distances_tr = np.empty((df_pairs_val.shape[0], 1))

if(iterations == 1): #case 1: of pairs_df less than 2500 rows
    left_imgs, right_imgs = load_pairs(df_pairs_val, img_filenames, imgs_processed)
    distances_tr = model.predict([left_imgs, right_imgs])
else: #case 2: of pairs_df more or equal to 2500 rows
    for i in range(iterations):
        print('STEP7: compute distance | iteration ' + str(i))
        if(i == 0):
            pairs_sub_df = df_pairs_val.iloc[0:max_batch, ]
            client_imgs, competitor_imgs = load_pairs(pairs_sub_df, img_filenames, imgs_processed)
            distances_tr_batch = model.predict([client_imgs, competitor_imgs])
            distances_tr[0:max_batch] = distances_tr_batch
        else: 
            pairs_sub_df = df_pairs_val.iloc[(i * max_batch):((i + 1) * max_batch), ]    
            client_imgs, competitor_imgs = load_pairs(pairs_sub_df, img_filenames, imgs_processed)
            distances_tr_batch = model.predict([client_imgs, competitor_imgs])
            distances_tr[(i * max_batch):((i + 1) * max_batch)] = distances_tr_batch




################################################################ SAVE OUTPUTS #####


print('STEP8: save objects' + run)
model.save(os.getcwd() + '/' + 'model_deepfashion88k_' + run + '.h5') 

with open(os.getcwd() + '/' + 'model_history_' + run + '.pickle', 'wb') as f:
    pickle.dump([history.history], f)    

df_pairs_val['distance_untr'] = distances_untr
df_pairs_val['distance_tr'] = distances_tr
df_pairs_val.to_csv(os.getcwd() + '/outputs/' + 'validation_pairs_' + run +'.csv', index = None)


###############################################################################

