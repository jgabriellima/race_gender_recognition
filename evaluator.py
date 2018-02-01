import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time


from multitask_model import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim

import glob
import os

import inception_preprocessing
from inception_resnet_v1 import inception_resnet_v1, inception_resnet_v1_arg_scope

from multitask_model import  *

data_file = './data/validate.tfrecords'
image_size = 200
num_races = 5

# classification parameter
minsize = 20 # minimum size of face
threshold = [0.7, 0.7]  #
# factor = 0.709 # scale factor

model_dir='./log/model_0/'
list_of_files = glob.glob(model_dir + '*') # * means all if need specific format then *.csv
latest_checkpoint = model_dir + 'model_iters_final'

log_dir = './log/model_0/'

#State the number of epochs to evaluate
batch_size = 32
num_epochs = 1

num_samples = 57234
num_batches_per_epoch = int(num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed

def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    # with tf.Graph().as_default() as graph:

    tf.reset_default_graph()

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    # create the dataset and load one batch
    images, genders, races = read_and_decode(data_file, [image_size, image_size], is_training=False)

    # build the multitask model
    logits_gender, logits_race, end_points, variables_to_restore = build_model(images)

    loss_genders = losses(logits_gender, slim.one_hot_encoding(genders, 2))
    loss_races = losses(logits_race, slim.one_hot_encoding(races, num_races))

    # Create the train_op.
    # loss = loss_genders + loss_races
    # loss = loss_genders
    end_points['Predictions/gender'] = tf.nn.softmax(logits_gender, name='Predictions/gender')
    end_points['Predictions/race'] = tf.nn.softmax(logits_race, name='Predictions/race')
    predictions1 = tf.argmax(end_points['Predictions/gender'], 1)
    predictions2 = tf.argmax(end_points['Predictions/race'], 1)

    accuracy1 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions1), genders)))
    accuracy2 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions2), races)))

    global_step = get_or_create_global_step()
    global_step_op = tf.assign(global_step,
                               global_step + 1)  # no apply_gradient method so manually increasing the global_step

    # Create a evaluation step function
    def eval_step(sess, global_step, summary_op):
        '''
        Simply takes in a session, runs the metrics op and some logging information.
        '''
        start_time = time.time()
        global_step_count, v1, v2, curr_summary = sess.run([global_step_op, accuracy1, accuracy2, summary_op])
        time_elapsed = time.time() - start_time

        # Log some information
        logging.info('Global Step %s: gender Accuracy: %.4f',
                     'race Accuracy: %.4f',
                     ' (%.2f sec/step)', global_step_count, v1, v2,
                     time_elapsed)

        return accuracy_value1, accuracy_value2, curr_summary

    tf.summary.scalar('Validation_Accuracy gender', accuracy1)
    tf.summary.scalar('Validation_Accuracy race', accuracy2)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess.run(init_op)
        saver.restore(sess, latest_checkpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())



        for step in range(num_steps_per_epoch):
            accuracy1, accuracy2, curr_summary = eval_step(sess, global_step, summary_op)
            writer.add_summary(curr_summary, step)


            if step % 100 == 0:
                print("Step%03d: " % (step + 1))

                logging.info('Current gender Accuracy: %s', accuracy1)
                logging.info('Current race Accuracy: %s', accuracy2)

            sess.run(global_step)
            accuracy_value1, accuracy_value2 = sess.run([accuracy1, accuracy2])

            logging.info('Current gender Accuracy: %s', accuracy_value1)
            logging.info('Current race Accuracy: %s', accuracy_value2)


        coord.request_stop()
        coord.join(threads)

        # saver.save(sess, final_checkpoint_file)
        sess.close()


