import tensorflow as tf
import numpy as np, os, sys

from auto_type import image_loader as il
from auto_type.main import predict_cls_test, cls_accuracy
from collections import Counter

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

sys.dont_write_bytecode = True

def init_variables():
    session.run(tf.global_variables_initializer())

x = tf.placeholder(tf.float32, shape=[None, il.img_size_flat], name='x')
x_image = tf.reshape(x, [-1, il.image_height, il.image_width, il.num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, il.num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
x_pretty = pt.wrap(x_image)

with tf.Graph().as_default(), pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=36, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=64, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=il.num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()

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


def predict_one_image(images):
    
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, il.num_classes),
                       dtype=np.float)
    
    # Create a feed-dict with the images between index i and j.
    feed_dict = {x: images[0:1, :]}

    # Calculate the predicted labels using TensorFlow.
    pred_labels[0:1] = session.run(y_pred, feed_dict=feed_dict)
    
    return pred_labels

def get_prediction(images_array):
    results = []
    results.append(np.argmax(predict_one_image(images_array[0:1,:])))
    results.append(np.argmax(predict_one_image(images_array[1:2,:])))
    results.append(np.argmax(predict_one_image(images_array[2:3,:])))
    results.append(np.argmax(predict_one_image(images_array[3:4,:])))
    resultsCounter = Counter(results)
    
    if results.count(2) >= 2:
        value = 2
    elif results.count(1) >= 2:
        value = 1
    else:
        value, _ = resultsCounter.most_common()[0]
    
    return value
    
init_variables()

saver = tf.train.Saver()

save_dir = 'checkpoints_final/'
save_path = os.path.join(save_dir, 'best_validation')

saver.restore(sess=session, save_path=save_path)

image_file = "arc.fits"
images_array = il.load_images_arr(image_file)
prediction = get_prediction(images_array)
print(il.Frames(prediction).name)
