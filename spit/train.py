""" Main methods for SPIT"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import tensorflow as tf
import numpy as np
import datetime
import os, sys, time
import pdb
import math

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

# Some globals

# The image batches that will be used in training sets
train_batch_size = 30

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 10

# Best validation accuracy seen so far.
best_val_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
require_improvement = 10000

# Counter for total number of iterations performed so far.
total_iterations = 0

def train_model(epochs, batch_size, subset_percent=None, train_images=None, train_labels=None, validation_data=None, steps_per_epoch=None, validation_freq=1, test_model=None, spit_path=os.getenv('SPIT_PATH'), save_path=os.getenv('SAVE_PATH')):
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
      model = test_model
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
    checkpoint=keras.callbacks.ModelCheckpoint(save_path+'best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

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
    
    # save loss and accuracy data
    keys = history.history
    np.savez_compressed('history.npz', loss=keys['loss'], accuracy=keys['accuracy'], val_loss=keys['val_loss'], val_accuracy=keys['val_accuracy'])
    
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


##################################################
# HELPER FUNCTION FOR CALCULATING CLASSIFICATIONS
##################################################

def predict_cls_wrap(classifier, images):
    """  Run the Classifier on input images
    Parameters
    ----------
    classifier : Classifier

    Returns
    -------

    """
    #from spit.preprocess import load_linear_pngs
    return predict_cls(classifier, images.images, images.labels, images.cls)


def predict_cls(classifier, images, labels, cls_true):
    """
    Parameters
    ----------
    classifier : Classifier
    images : list
    labels : list
    cls_true

    Returns
    -------

    """
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {classifier.x: images[i:j, :],
                     classifier.y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = classifier.session.run(classifier.y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def random_train_batch(images_train):
    # Number of images in the training-set.
    num_images = images_train.images.shape[0]

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train.images[idx, :]
    y_batch = images_train.labels[idx, :]
    
    return x_batch, y_batch

#####################################
# FUNCTONS TO CALCULATE THE ACCURACY
#####################################
def accuracy(classifier, images):
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_wrap(classifier, images)
    
    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)


def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

#####################################
# PRINT TEST ACCURACIES
#####################################
def print_test_accuracy(classifier, images, show_example_errors=False,
                        show_confusion_matrix=False):
    """
    Parameters
    ----------
    classifier : Classifier
    show_example_errors : bool, optional
    show_confusion_matrix : bool, optional

    Returns
    -------

    """
    # Hide plot imports
    if show_example_errors or show_confusion_matrix:
        from spit.plots import plot_example_errors
        from spit.plots import plot_confusion_matrix

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, classifier.cls_pred = predict_cls_wrap(classifier, images)

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(images, cls_pred=classifier.cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(images, classifier)

########################
# OPTIMIZATION FUNCTION
########################
def optimize(classifier, images_train, images_val, save_validation_path, num_iterations=10):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_val_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_train_batch(images_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {classifier.x: x_batch,
                           classifier.y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        classifier.session.run(classifier.optimizer, feed_dict=feed_dict_train)

        # Calculate the accuracy on the training-batch.
        acc_train = classifier.session.run(classifier.accuracy, feed_dict=feed_dict_train)

        # Calculate the accuracy on the validation-set.
        # The function returns 2 values but we only need the first.
        #acc_validation, _ = validation_accuracy()
        acc_val, _ = accuracy(classifier, images_val)

        # If validation accuracy is an improvement over best-known.
        if acc_val > best_val_accuracy:
            # Update the best-known validation accuracy.
            best_val_accuracy = acc_val
                
            # Set the iteration for the last improvement to current.
            last_improvement = total_iterations

            # Save all variables of the TensorFlow graph to file.
            classifier.saver.save(sess=classifier.session, save_path=save_validation_path)

            # A string to be printed below, shows improvement found.
            improved_str = '*'
        else:
            # An empty string to be printed below.
            # Shows that no improvement was found.
            improved_str = ''
            
	#if total_iterations % 100 == 0:
	#	print("Accuracy on the test set: ")
	#	print_test_accuracy()
        # Status-message for printing.
        msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Val Acc: {2:>6.1%} {3}"

        # Print it.
        print(msg.format(i + 1, acc_train, acc_val, improved_str))
        print(str(datetime.datetime.now()))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(datetime.timedelta(seconds=int(round(time_dif)))))

def print_versions():
    print("Printing library versions")
    print("TensorFlow: " + tf.__version__)
    print("Prety Tensor: " + pt.__version__)
    print("Python: " + sys.version)

def run(instrument, num_iterations=10):
    from spit.images import Images
    from spit.classifier import Classifier
    print_versions()

    # Setup the TensorFlow model
    classifier = Classifier()

    # Load the dataset
    images_train = Images('Kast_train')
    images_test = Images('Kast_test')
    images_val = Images('Kast_validation')
    #images_train, cls_train, labels_train, filenames_train = .load_linear_pngs(
    #    data_type="train_data")
    #images_test, cls_test, labels_test, filenames_test = preprocess.load_linear_pngs(
    #    data_type="test_data")


    # Dataset sizes
    print("Size of:")
    print("- Training-set:\t\t{}".format(images_train.images.shape[0]))
    print("- Validation-set:\t{}".format(images_val.images.shape[0]))
    print("- Test-set:\t\t{}".format(images_test.images.shape[0]))


    ########################################################################

    # Saver Method to save the best performing NN
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_validation_path = os.path.join(save_dir, 'best_validation')
    save_done_path = os.path.join(save_dir, 'done_training')


    # Print test accuracy with no optimization
    print("Measuring test accuracy without optimization")
    print_test_accuracy(classifier, images_test)

    # Perform the optimization
    optimize(classifier, images_train, images_val, save_validation_path, num_iterations=num_iterations)

    # Print the test accuracy after 100 optimizations
    print("Accuracies after {:d} iterations!".format(num_iterations))
    print_test_accuracy(classifier, images_test)

    # Save the model after it's done training
    classifier.saver.save(sess=classifier.session, save_path=save_done_path)

    # Perform the optimization
    #optimize(num_iterations=9000)

    # Print the test accuracy after 100 optimizations
    #print("Accuracies after 10000 iterations!")
    #print_test_accuracy()

    # Initialize Variables Again
    print("Clearing Variables for untrained")
    classifier.init_variables()
    print_test_accuracy(classifier, images_test)

    # Restore best Variables
    print("Best case (test)")
    classifier.saver.restore(sess=classifier.session, save_path=save_validation_path)
    print_test_accuracy(classifier, images_test)

    # Restore done Variables
    classifier.saver.restore(sess=classifier.session, save_path=save_done_path)

    print("Last case [aka done] (test)!")
    print_test_accuracy(classifier, images_test)


    classifier.session.close()

# Command line execution
if __name__ == '__main__':
    run('Kast', num_iterations=1000)

