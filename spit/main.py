""" Main methods for SPIT"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import tensorflow as tf
import numpy as np
from datetime import timedelta
import os, sys, time

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

##################################################
# HELPER FUNCTION FOR CALCULATING CLASSIFICATIONS
##################################################
def predict_cls_validation():
    return predict_cls(images = images_val,
                       labels = labels_val,
                       cls_true = cls_val)

def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

def predict_cls(images, labels, cls_true):
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
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def random_train_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]
    
    return x_batch, y_batch

#####################################
# FUNCTONS TO CALCULATE THE ACCURACY
#####################################
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()
    
    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)

def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_test()
    
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
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

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
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

########################
# OPTIMIZATION FUNCTION
########################
def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_test_accuracy
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
        x_batch, y_true_batch = random_train_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Calculate the accuracy on the training-batch.
        acc_train = session.run(accuracy, feed_dict=feed_dict_train)

        # Calculate the accuracy on the validation-set.
        # The function returns 2 values but we only need the first.
        #acc_validation, _ = validation_accuracy()
        acc_test, _ =test_accuracy()

        # If validation accuracy is an improvement over best-known.
        if acc_test > best_test_accuracy:
            # Update the best-known validation accuracy.
            best_test_accuracy = acc_test
                
            # Set the iteration for the last improvement to current.
            last_improvement = total_iterations

            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=session, save_path=save_validation_path)

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
        msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Test Acc: {2:>6.1%} {3}"

        # Print it.
        print(msg.format(i + 1, acc_train, acc_test, improved_str))

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
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def print_versions():
    print("Printing library versions")
    print("TensorFlow: " + tf.__version__)
    print("Prety Tensor: " + pt.__version__)
    print("Python: " + sys.version)

def run():
    from spit import preprocess
    print_versions()

    # Load the dataset
    images_train, cls_train, labels_train, filenames_train = preprocess.load_linear_pngs(
        data_type="train_data")
    images_test, cls_test, labels_test, filenames_test = preprocess.load_linear_pngs(
        data_type="test_data")

    # Dataset sizes
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    #print("- Validation-set:\t{}".format(len(images_val)))
    print("- Test-set:\t\t{}".format(len(images_test)))


    ########################################################################
    # Various constants for the size of the images.

    # The height of an image
    image_height = preprocess.image_height

    # The width of an image
    image_width = preprocess.image_width

    # Length of an image when flattened to a 1-dim array.
    img_size_flat = image_height * image_width

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (image_height, image_width)

    # Number of channels in each image, 3 channels: Red, Green, Blue.
    num_channels = preprocess.num_channels

    # Number of classes.
    num_classes = preprocess.num_classes

    # The padding value for the padded image
    pad_const = preprocess.pad_const
    ########################################################################

    # Start building the TensorFlow model
    # Placeholder Variables
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, image_height, image_width, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # Neural Network
    x_pretty = pt.wrap(x_image)

    # Basic CNN architecture
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

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=36, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

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

    # Optimize Method
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Performance Measures
    y_pred_cls = tf.argmax(y_pred, axis=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Saver Method to save the best performing NN
    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_validation_path = os.path.join(save_dir, 'best_validation')
    save_done_path = os.path.join(save_dir, 'done_training')

    # The image batches that will be used in training sets
    train_batch_size = 10

    # Split the data-set in batches of this size to limit RAM usage.
    batch_size = 10

    # Best validation accuracy seen so far.
    best_test_accuracy = 0.0

    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0

    # Stop optimization if no improvement found in this many iterations.
    require_improvement = 10000

    # Counter for total number of iterations performed so far.
    total_iterations = 0

    # TensorFlow run
    session = tf.Session()

    def init_variables():
        session.run(tf.global_variables_initializer())

    init_variables()

    # Print test accuracy with no optimization
    print_test_accuracy()

    # Perform the optimization
    optimize(num_iterations=10000)

    # Print the test accuracy after 100 optimizations
    print("Accuracies after 1000 iterations!")
    print_test_accuracy()

    # Save the mode lafter it's done training
    saver.save(sess=session, save_path=save_done_path)

    # Perform the optimization
    #optimize(num_iterations=9000)

    # Print the test accuracy after 100 optimizations
    #print("Accuracies after 10000 iterations!")
    #print_test_accuracy()

    # Initialize Variables Again
    print("Clearing Variables")
    init_variables()
    print_test_accuracy()

    # Restore best Varibles
    saver.restore(sess=session, save_path=save_validation_path)

    print("Print Validation case accuracies!")
    print_test_accuracy()
    init_variables()
    print(test_accuracy())

    # Restore best Varibles
    saver.restore(sess=session, save_path=save_done_path)

    print("Print Validation case accuracies!")
    print_test_accuracy()


    session.close()
