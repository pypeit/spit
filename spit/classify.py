""" Module to run classification of images """
import numpy as np, os, sys
import pdb

from spit import image_loader as spit_il
from spit import defs
from collections import Counter

# Use PrettyTensor to simplify Neural Network construction.

sys.dont_write_bytecode = True


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
    pred_labels = np.zeros(shape=(num_images, defs.num_classes),
                       dtype=np.float)
    
    # Create a feed-dict with the images between index i and j.
    feed_dict = {classifier.x: images[0:1, :]}

    # Calculate the predicted labels using TensorFlow.
    pred_labels[0:1] = classifier.session.run(classifier.y_pred, feed_dict=feed_dict)
    
    return pred_labels


def get_prediction(images_array, classifier, use_heuristics=False):
    """

    Returns
    -------
    value : int
      Most common value or -1, if it is not the majority
    results : list
      All of the values for each flipped image
    """
    # Classify all 4
    results = []
    results.append(np.argmax(predict_one_image(images_array[0:1,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[1:2,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[2:3,:], classifier)))
    results.append(np.argmax(predict_one_image(images_array[3:4,:], classifier)))
    resultsCounter = Counter(results)

    # Heuristics
    if use_heuristics:
        if results.count(2) >= 2:
            value = 2
        elif results.count(1) >= 2:
            value = 1
        else:
            value, _ = resultsCounter.most_common()[0]
    else:  # Majority rules
        value, n_occur = resultsCounter.most_common()[0]
        if n_occur <= 2:
            value = -1
    # Return
    return value, results


def classify_me(image_file, classifier_root=None, verbose=False):
    from spit.classifier import Classifier

    # Generate a Classifier
    classifier = Classifier(classifier_root)

    # Image array (4 flips)
    images_array = spit_il.load_images_arr(image_file)

    # Predict
    prediction, results = get_prediction(images_array, classifier)
    if verbose:
        print("Input image {:s} is classified as a {:s}".format(image_file,
                                                                spit_il.Frames(prediction).name))

    # Return
    return spit_il.Frames(prediction).name, results



