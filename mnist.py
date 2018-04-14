import tensorflow as tf
import tensorflow.contrib.slim as slim # Simplified TensorFlow
import matplotlib.pyplot as plt
import numpy as np
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import matplotlib.colors as colors
color_list = list(colors.cnames.values())


def outPossibilities(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plt.figure()
    g = np.arange(10)
    plt.bar(g, height=np.ravel(units), width=1)
    plt.xticks(g+.5, range(10));
    plt.suptitle("Output", fontsize=16, fontweight='bold')
    plt.xlabel("Written number")
    plt.ylabel("Possibility")
    plt.show(block=False)


def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(layer, units)


def plotNNFilter(layer, units):
    filters = str(units.shape[3])
    s = str(units.shape[2])
    plt.figure()

    for i in range(9):
        ax = plt.subplot(3, 3 , i+1)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

    txt = "Layer size: [" + s + ", " + s + ", " + filters + "]"
    plt.suptitle(txt, fontsize=16, fontweight='bold')
    plt.show(block=False)


def plot_img(image, label):
    two_d = (np.reshape(image, (28, 28))* 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest' , cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
    plt.suptitle("Label: " + str(label.tolist().index(1)),fontsize=16, fontweight='bold')
    plt.show(block=False)


def plotGraph(title, *args):
    plt.figure()
    plt.suptitle(title, fontsize=18, fontweight='bold')
    for i, list in enumerate(args):
        plt.plot(list[0], color=color_list[i], label=list[1], linewidth=2)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    image_for_display = mnist.test.images[3]
    label_for_display = mnist.test.labels[3]

    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 784]) # 28 x 28
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    xr = tf.reshape(x, [-1, 28, 28, 1])

    hidden_1 = slim.conv2d(xr,24,[1,1], activation_fn=tf.nn.elu, normalizer_fn=slim.batch_norm)
    hidden_1_drop = slim.dropout(hidden_1,keep_prob)
    pool_1 = slim.max_pool2d(hidden_1_drop,[2,2])
    hidden_2_drop = slim.conv2d(pool_1,200,[4,4], activation_fn=tf.nn.elu, normalizer_fn=slim.batch_norm)
    pool_2 = slim.max_pool2d(hidden_2_drop,[2,2])
    hidden_3 = slim.conv2d(pool_2,20,[4,4], activation_fn=tf.nn.elu, normalizer_fn=slim.batch_norm)
    hidden_3_drop = slim.dropout(hidden_3,keep_prob)
    output = slim.fully_connected(slim.flatten(hidden_3_drop),10, activation_fn=tf.nn.softmax, normalizer_fn=slim.batch_norm)

    prediction = tf.nn.softmax(output) # Format for loss check

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess.run(tf.global_variables_initializer())

    #plot_img(image_for_display, label_for_display) # Plot input data

    print "-----------------------------------"

    samples = 55000
    batch_size = 100
    epochs = 5
    validation_samples = 5000

    loss_val_list = []
    acc_val_list = []

    loss_list = []
    acc_list = []

    for i in range(epochs):
        for j in range(samples/batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs -= 0.5

            _, loss, acc = sess.run([train, loss_op, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            acc_list.append(acc)
            loss_list.append(loss)

            if (j % 25 == 0):
                print 'Epoch: {0:3}/{1} Sample nr.: {2:5}/{3} Loss: {4:.3f} Acc.: {5:.3f}'.format(i+1, epochs, j*batch_size, samples, loss, acc)

        for k in range(validation_samples/epochs/batch_size):
            batch_xs_val, batch_ys_val = mnist.validation.next_batch(batch_size)
            batch_xs_val -= 0.5

            loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={x: batch_xs_val, y: batch_ys_val, keep_prob: 1.0})
            loss_val_list.append(loss_val)
            acc_val_list.append(acc_val)

        print 'Validation of epoch nr. {0}/{1} is done. Val. loss: {2:.5f} Acc.: {3:.5f}'.format(i+1, epochs, np.mean(loss_val_list), np.mean(acc_val_list))



    print "\nTraining finished!\n-----------------------------------"


    final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:501],y: mnist.test.labels[:501], keep_prob: 1.0})
    print "Testing Accuracy:", final_acc

    plotGraph('Loss and Accuracy', [loss_list, 'Loss'], [acc_list, 'Acc.'])
    plotGraph('Validation - Loss and Accuracy', [loss_val_list, 'Val. loss'], [acc_val_list, 'Val. acc.'])
    getActivations(hidden_1, image_for_display)
    getActivations(pool_1, image_for_display)
    getActivations(hidden_2, image_for_display)
    getActivations(pool_2, image_for_display)
    getActivations(hidden_3, image_for_display)
    outPossibilities(output, image_for_display)

    plt.ion()
    plt.show()

    raw_input("Press Enter to continue...")
