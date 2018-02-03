""" SPIT Classifier object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
import numpy as np
from pkg_resources import resource_filename

from matplotlib import pyplot as plt


def plot_images(images, cls_true, filenames, cls_pred=None, img_height=210, image_width=650):
    assert len(images) == len(cls_true) == len(filenames)
    # Init
    img_shape = (image_height, image_width)

    # Create figure with 2x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.set_size_inches(18.5, 10.5)

    files = []
    for f in filenames:
        file = f.split("/")
        file = file[-2] + "_" + file[-1]
        files.append(file)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if cls_pred is None:
                if i == 1 or i == 4 or i == 7:
                    xlabel = "True: {0}, Fn:\n\n\n{1}".format(cls_true[i], files[i])
                else:
                    xlabel = "True: {0}, \nFn: {1}".format(cls_true[i], files[i])
            else:
                if i == 1 or i == 4 or i == 7:
                    xlabel = "True: {0}, Pred: {1}, Fn:\n\n\n{2}".format(cls_true[i], cls_pred[i], files[i])
                else:
                    xlabel = "True: {0}, Pred: {1}, \nFn: {2}".format(cls_true[i], cls_pred[i], files[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(images, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    bad_images = images.images[incorrect]

    # Get the filenames
    filenames = []
    for i in bad_images:
        index = np.where((images.images == i).all(axis=1))[0][0]
        filename = images.filenames[index]
        filenames.append(filename)

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = images.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=bad_images[0:9],
                cls_true=cls_true[0:9],
                filenames=filenames[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(images, classifier):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    from sklearn.metrics import confusion_matrix
    from spit import preprocess
    num_classes = preprocess.num_classes

    cls_true = images.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=classifier.cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

