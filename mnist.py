import tensorflow as tf
import numpy as np
import math

import tensorflow.contrib.slim as slim

#import matplotlib.pyplot as plt

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv2d(x, W, s, name):
	return tf.nn.conv2d(x, W, strides=[1,s,s,1], padding="SAME",name=name)

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variabel(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(layer, units)

def plotNNFilter(layer, units):

    filters = units.shape[3]
    plt.figure() # par:
    n_columns = 6
    print "STEEEEEk:", filters

    n_rows = math.ceil(filters / n_columns) + 1

    for i in range(filters):
        ax = plt.subplot(n_rows, n_columns ,i+1)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

    plt.suptitle(str(layer), fontsize=8, fontweight='bold')
    plt.show(block=False)


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        arr = images
        arr = arr[i,:]
        two_d = (np.reshape(arr, (28, 28))* 255).astype(np.uint8)
        ax.imshow(two_d, interpolation=interpolation , cmap=plt.get_cmap('gray')) # cmap='binary'

        cls_true_name = cls_true[i]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


def plot_layer_output(layer_output, image):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {x: [image]}

    # Retrieve the output of the layer after inputting this image.
    values = sess.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    cols =  num_images / int(num_grids)

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(int(num_grids), cols)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i<num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap=plt.get_cmap('gray'))

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show(block=False)

def plot_img(image):

    two_d = (np.reshape(image, (28, 28))* 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest' , cmap=plt.get_cmap('gray'))

    plt.tick_params(axis=u'both', which=u'both',length=0)

    #plt.imshow(image, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
    #plt.set_xticks([])
    #plt.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show(block=False)


def get_layer_output(layer_name):

    tensor = tf.get_default_graph().get_tensor_by_name(layer_name)

    return tensor


if __name__ == "__main__":

    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 784]) # 28 x 28
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    xr = tf.reshape(x, [-1, 28, 28, 1])

    hidden_1 = slim.conv2d(xr,5,[5,5])
    pool_1 = slim.max_pool2d(hidden_1,[2,2])
    hidden_2 = slim.conv2d(pool_1,5,[5,5])
    pool_2 = slim.max_pool2d(hidden_2,[2,2])
    hidden_3 = slim.conv2d(pool_2,20,[5,5])
    hidden_3 = slim.dropout(hidden_3,keep_prob)
    output = slim.fully_connected(slim.flatten(hidden_3),10,activation_fn=tf.nn.softmax)




    # conv1
    #W_conv1 = weight_variable([4,4,1,20]) # [filter height,filter height,input depth,output depth]
    #b_conv1 = bias_variabel([20]) # [output depth]
    #h_conv1 = tf.nn.elu(conv2d(xr, W_conv1, s=1, name='layer_conv1') + b_conv1)

    #elu1 = tf.nn.elu(h_conv1)
    #pool1 = maxpool2d(elu1, 2)


    # conv2
    #W_conv2 = weight_variable([5,5,20,32])
    #b_conv2 = bias_variabel([32])
    #h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, s=1,name='layer_conv2') + b_conv2)

    # drop1
    #drop1 = tf.nn.dropout(h_conv2, keep_prob)

    # elu1
    #elu1 = tf.nn.elu(drop1)

    #pool2 = maxpool2d(elu1, 2)

    #fc1 = tf.contrib.layers.flatten(pool2)
    #den = tf.layers.dense(fc1, 1024)

    #elu2 = tf.nn.elu(den)

    #output = tf.layers.dense(elu2, 10) # Output

    prediction = tf.nn.softmax(output) # Format for loss check

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess.run(tf.global_variables_initializer())

    plt.ioff()
    #plot_img(mnist.test.images[0]) # Print input data

    print "-----------------------------------------------"
    print [n.name for n in tf.get_default_graph().as_graph_def().node]
    print "-----------------------------------------------"

    for i in range(40):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.9})
        if (i % 10 == 0):
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            print 'Epoch: {0:3} Loss: {1:.3f} Acc.: {2:.3f}'.format(i, loss, acc)

    print "\nTraining finished!\n-----------------------------------"

    final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256], keep_prob: 1.0})
    print "Testing Accuracy:", final_acc

    image = mnist.test.images[0]

    #output_conv1 = get_layer_output(layer_name="layer_conv1:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="MaxPool:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="layer_conv2:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="MaxPool_1:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="dense_1:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="dense_2:0")
    #plot_layer_output(output_conv1, image)

    #output_conv1 = get_layer_output(layer_name="Softmax:0")
    #plot_layer_output(output_conv1, image)

    #arr = image
    #arr = arr[i,:]
    #two_d = (np.reshape(arr, (28, 28))* 255).astype(np.uint8)

    #getActivations(h_conv1, image)

    getActivations(hidden_2, image)

    #getActivations(drop1, image)

    #getActivations(pool2, image)

    #getActivations(output, image)

    plt.ion()
    plt.show()
    raw_input("Press Enter to continue...")
