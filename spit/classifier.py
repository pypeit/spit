""" New SPIT Classifier object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
import numpy as np
from spit import preprocess
from spit import labels
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import math

class Classifier(object):


  # For Kast, label_dict = labels.kast_label_dict()
  # preproc_dict = preprocess.original_preproc_dict()
  # classify_dict = labels.kast_classify_dict(label_dict)
  def __init__(self, label_dict, preproc_dict, classify_dict, **kwargs):
    self.label_dict = label_dict.copy()
    self.preproc_dict = preproc_dict.copy()
    # is this for prediction? <----
    self.classify_dict = classify_dict.copy() 
    
    # Set up tensorflow/keras model
    self.init_session()
    
  
  # One major change from tf1.0 to tf2.0
  # is that tf.layers module no longer relies on tf.variable_scope
  # and no more use of tf.placeholder
  # no more session.run() although not actual functioning originally 
  def init_session(self):
    
    # building a simple model
    # linear stack of layers

    self.model = keras.Sequential([
        # 2D convolution followed by a maxpool, change data format when actual dataset etc. comes or when testing
        keras.layers.Conv2D(36, kernel_size=5, strides=(1, 1), padding='valid', activation='relu', 
                            input_shape = (self.preproc_dict['image_height'], self.preproc_dict['image_width'], self.preproc_dict['num_channels'])),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        # And another, this time with 64 filters instead of 36
        keras.layers.Conv2D(64, kernel_size=5, strides=(1, 1), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        # Flatten in preparation for FC layer
        keras.layers.Flatten(),
        # FC layer
        keras.layers.Dense(units=128, activation='relu'),
        # Produce 0-1 probabilities with softmax
        keras.layers.Dense(len(self.label_dict), activation='softmax')
    ])
    # convert labels to respective categories for training
    self.y_train = keras.utils.to_categorical(list(self.label_dict.values()), 
                                              num_classes=len(self.label_dict))
    # add optimizer, learning rate, and loss function
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])#, rate=1e-4)
    return

  def load_model(self, file_name, file_path):
    '''

    :param file_name: The model name. Should be a string 'blah.h5'
           file_path: The path that the user would like to save to.
                      In the form of 'blah/foo/bar/'
    :return: the model loaded
    '''

    loaded_model = keras.models.load_model(file_path+file_name)
    return loaded_model

  def save_model(self, model_to_save, file_name, file_path):
    '''

    :param file_name: The model name. Should be a string 'blah.h5'
           file_path: The path that the user would like to save to.
                      In the form of 'blah/foo/bar/'
    :return: N/A
    '''

    model_to_save.save(file_path+file_name)
    return
  
  def evaluate(self, test_images, test_labels, test_model=None):
    """
    Evaluate the model on an unseen dataset.
    Assume the model is constructed and trained already.

    :param test_model:
        An alternative choice for model to test.
    :param test_images:
        Set of test images not seen by the model yet.
        Assume this is a numpy array with (batch_size, width, height, num_channels) as its dimensions.
    :param test_labels:
        Set of test labels corresponding to test images.
        Assume this is a vector with (batch_size, 1) as its dimensions.

    :return:
      loss: Loss of the evaluated model as a float value.
      acc: Accuracy of the evaluated model as a float value between 0 and 1.
    """

    # make categorical for model
    y_test = keras.utils.to_categorical(test_labels, num_classes=len(self.label_dict))
    # evaluate model
    if test_model is None:
      loss, acc = self.model.evaluate(test_images, y_test)
    else:
      loss, acc = test_model.evaluate(test_images, y_test)
    # return loss and accuracy as array
    return loss, acc
  
  def compare_with_best(self, test_dset, test_labels, file_path):
    '''
    Method to compare current model with the best_model and save a new best_model
    :param test_dset: test data
    :param test_labels: test labels
    :param file_path: where to save the model. In the form 'blah/foo/bar/'
    :return: N/A
    '''

    try:
      best_model = self.load_model('best_model.h5', file_path)
      has_best = True
    except:
      has_best = False
      pass

    if has_best:
      loss_self, acc_self = self.evaluate(test_dset, test_labels)
      loss_best, acc_best = self.evaluate(test_dset, test_labels, test_model=best_model)
      if acc_self > acc_best:
        self.save_model(self.model, 'best_model.h5', file_path)
        del best_model
      else:
        del best_model
    else:
      self.save_model(self.model, 'best_model.h5', file_path)

    return

  def _train(self, epochs, batch_size, subset_percent=None, train_images=None, train_labels=None, validation_data=None, steps_per_epoch=None, validation_freq=1, test_model=None, spit_path=os.getenv('SPIT_PATH'), save_path=os.getenv('SAVE_PATH')):
    """

    Trains the classifier with given images, labels, and training parameters.

    Parameters:

    :param epochs:
      Number of epochs of the training. 
      Must be an integer value.

    :param batch_size:
      Size of the training batches formed in the process. 
      Must be an integer value.

    :param save_path:
      Path to where the best model will be saved.

    :param train_images:
       Set of images for model to train on.
       Assume this is a numpy array with (batch_size, width, height, num_channels) as its dimensions.

    :param train_labels:
      Set of test labels corresponding to test images.
      Assume this is a rank 1 array with (batch_size, ) as its dimensions.

    :param validation_data:
      Data to be used for the validation set. 
      Assume this is a tuple with (images, labels) with same dimensions as train_images, train_labels.
      If None is specified, validation_data will be None.

    :param steps_per_epoch:
      Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
      Assume this is an integer value. If None is specified, this will be None.  

    :param validation_freq:
      Specifies how many training epochs to run before a new validation run is performed.
      Assume this is an integer value or a collection containing the epochs at which to run validation (ie [1,2,10]).

    :param test_model:
      An alternative choice for model to train on. Assume None.

    :param spit_path & param save_path:
      Path to the spit images and where the model will be saved respectively.
      ***Environmental variables must be set by caller or the path must be passed manually.***

    Returns:
    :returns history:
      Tensorflow History object containing loss and accuracy data over the training.

    """
    # choose the model
    if test_model is None:
      model = self.model
    else:
      model = test_model

    # if None is passed, then use the kast images
    if train_images is None or train_labels is None:
      # load training set
      train = np.load(os.path.join(spit_path, 'Kast', 'kast_train.npz'))
      train_images = train['images']
      train_labels = train['labels']

      # load validation set
      validate = np.load(os.path.join(spit_path, 'Kast', 'kast_validate.npz'))
      v_images = validate['images']
      v_labels = validate['labels']

      validation_data = (v_images, v_labels)

    # change to categorical and make subsets
    if validation_data is not None:
      valid_images, valid_labels = validation_data
      if subset_percent is not None:
        valid_images, valid_labels = split_array(valid_images, valid_labels, subset_percent)
      valid_labels = keras.utils.to_categorical(valid_labels, num_classes=len(label_dict))
      validation_data = (valid_images, valid_labels)

    if train_images is not None and train_labels is not None:
      if subset_percent is not None:
        train_images, train_labels = split_array(train_images, train_labels, subset_percent)
      train_labels = keras.utils.to_categorical(train_labels, num_classes=len(label_dict))

    # checkpoint to track best model
    checkpoint=keras.callbacks.ModelCheckpoint(save_path+'best_model.h5', monitor='val_acc', save_best_only=True, mode='max')

    # train the model
    history = model.fit(
          train_images, 
          train_labels, 
          epochs=epochs, 
          batch_size=batch_size,
          validation_data=validation_data,
          steps_per_epoch=steps_per_epoch,
          validation_freq=validation_freq,
          callbacks=[checkpoint]
    )
    # loss and accuracy data
    keys = history.history
    # save to disc differently based on whether validation set was used
    if len(history.history.keys()) == 2: # can we abstract this away?
      np.savez_compressed('history.npz', loss=keys['loss'], acc=keys['acc']) #can these keys be abstracted away?
    else:
      np.savez_compressed('history.npz', loss=keys['loss'], acc=keys['acc'], val_loss=keys['val_loss'], val_acc=keys['val_acc'])

    return history
  
  def split_array(images, labels, subset_percent):
    """
    Splits dataset based on a percentage value.

    Parameters:
    :param images:
      Images from a dataset to be trained on.
      4-D Numpy array with (batch_size, width, height, num_channels) as its dimensions.

    :param labels:
      Labels from a dataset to be trained on.
      Rank 1 Numpy array with (batch_size,) as its dimensions.

    :param subset_percent:
      Float value determining percentage of subset to remain.

    Returns:
    :returns split_images:
      Numpy array containing a fraction of the initial images parameter (batch_size*subset_percent)

    :returns split_labels:
      Numpy array containing a fraction of the initial labels parameter (batch_size*subset_percent)
    """
    split_images = []
    split_labels = []
    
    # get all unique labels
    uni_lbls = np.unique(labels)
    
    # find all instances of labels and subset based on that
    for uni_lbl in uni_lbls:
      idx = np.where(labels==uni_lbl)[0]
      # 0 : len(idx)*subset_percent
      lower = 0
      upper = int(math.floor(len(idx)*subset_percent))
      split_images.extend(images[idx[lower:upper]])
      split_labels.extend(labels[idx[lower:upper]])
    return np.asarray(split_images), np.asarray(split_labels)

    # Other architectures
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=36, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)
    """

    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=36, name='layer_conv1').\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            conv2d(kernel=5, depth=72, name='layer_conv2').\
            conv2d(kernel=5, depth=102, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)
    """

    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)
    """
