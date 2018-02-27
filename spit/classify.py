import tensorflow as tf
import numpy as np, os, sys

from spit import image_loader as il
from spit.train import predict_cls, cls_accuracy
from collections import Counter

# Use PrettyTensor to simplify Neural Network construction.

sys.dont_write_bytecode = True


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))


def predict_one_image(images, classifier):
    """
    Parameters
    ----------
    images : list
    classifier : Classifier

    Returns
    -------
    pred_labels :

    """
    
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, il.num_classes),
                       dtype=np.float)
    
    # Create a feed-dict with the images between index i and j.
    feed_dict = {classifier.x: images[0:1, :]}

    # Calculate the predicted labels using TensorFlow.
    pred_labels[0:1] = classifier.session.run(classifier.y_pred, feed_dict=feed_dict)
    
    return pred_labels

def get_prediction(images_array, classifier):
    results = []
    results.append(np.argmax(predict_one_image(images_array[0:1,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[1:2,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[2:3,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[3:4,:], classifier)))
    resultsCounter = Counter(results)
    
    if results.count(2) >= 2:
        value = 2
    elif results.count(1) >= 2:
        value = 1
    else:
        value, _ = resultsCounter.most_common()[0]
    
    return value

def classify_me(image_file, classifier_root=None, verbose=False):
    from spit.classifier import Classifier

    # Generate a Classifier
    classifier = Classifier(classifier_root)

    # Image array
    images_array = il.load_images_arr(image_file)

    # Predict
    prediction = get_prediction(images_array, classifier)
    if verbose:
        print("Input image {:s} is classified as a {:s}".format(image_file, il.Frames(prediction).name))

    # Return
    return il.Frames(prediction).name


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(cls)-1.
    :return:
        2-dim array of shape: [len(cls), num_classes]
    """
    # Find the number of classes if None is provided.
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]
