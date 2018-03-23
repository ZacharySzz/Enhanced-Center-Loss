import os
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
k = 1 # for harmonic mean
threshold = 0.4
range_val = 2
LAMBDA = 1.0
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 10
slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
epoch = 0
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None,28,28,1), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    
global_step = tf.Variable(0, trainable=False, name='global_step')

def get_center_loss(features, labels, alpha, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(features - centers_batch)
    
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    pairwise_differences = features[:, tf.newaxis] - centers_batch[tf.newaxis, :]
    pairwise_differences_shape = tf.shape(pairwise_differences)
    
    mask = 1 - tf.eye(pairwise_differences_shape[0], pairwise_differences_shape[1], dtype=diffs.dtype)
    pairwise_differences = pairwise_differences * mask[:, :, tf.newaxis]
    
    # computing the loss
    pairwise_loss = tf.square(tf.nn.l2_loss(pairwise_differences)) # squaring the pairwise_loss
    loss  = (loss) + (k / pairwise_loss ) # harmonic mean of pairwise loss 


    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    
    return loss, centers_update_op


def inference(input_images):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            
            x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')
     
            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')
            
            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')
            
            x = slim.flatten(x, scope='flatten')
            
            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')
            
            x = tflearn.prelu(feature)

            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
    
    return x, feature

def build_network(input_images, labels, ratio=0.5):
    logits, features = inference(input_images)
    
    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss
    
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
    
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
        
    return logits, features, total_loss, accuracy, centers_update_op # returns total loss

logits, features, total_loss, accuracy, centers_update_op = build_network(input_images, labels, ratio=LAMBDA)
mnist = input_data.read_data_sets('/tmp/mnist', reshape=False)
optimizer = tf.train.AdamOptimizer(0.001)
with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
save_path = saver.save(sess, "/home/super/PycharmProjects/Recursive-CenterLoss/Enhanced_CenterLoss/model-maxi/model-maxi/model.ckpt")
print("Model saved in path: %s" % save_path)
writer = tf.summary.FileWriter('/tmp/mnist_log', sess.graph)
mean_data = np.mean(mnist.train.images, axis=0)
step = sess.run(global_step)
while step <= 8000:
    batch_images, batch_labels = mnist.train.next_batch(128)
    _, summary_str, train_acc, train_loss = sess.run(
        [train_op, summary_op, accuracy, total_loss],
        feed_dict={
            input_images: batch_images - mean_data,
            labels: batch_labels,
        })
    step += 1
    writer.add_summary(summary_str, global_step=step) 
    print(("Step: {}, Loss: {:.4f}".format(step, train_loss))) #prints training loss and steps. 
    if step % 200 == 0:
        epoch += 1
        vali_image = mnist.validation.images - mean_data
        vali_acc = sess.run(
            accuracy,
            feed_dict={
                input_images: vali_image,
                labels: mnist.validation.labels
            })
        print("\nEpochs Done: {}".format(epoch))
        print("{} Steps Done.".format(step))
        
        print(("Step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
              format(step, train_acc, vali_acc)))
        print("\n")
        print("====================================================")
        print("The value of centers: ")
        li = (sess.run(centers_update_op, feed_dict={
            input_images: batch_images - mean_data,
            labels: batch_labels,
        }))
        n = [0,1,2,3,4,5,6,7,8,9]
        print("Plot at epoch # {}".format(epoch))
        fig, ax = plt.subplots()
        for i, txt in enumerate(n):
            x,y = (li[i][0], li[i][1])
            ax.annotate(txt, (x,y))
            plt.scatter(x,y)
            fig.savefig('PlotWithnewtest-{}.png'.format(epoch))
        print("====================================================")
        print("\n")
