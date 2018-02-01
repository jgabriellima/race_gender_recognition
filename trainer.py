import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from inception_resnet_v1 import inception_resnet_v1, inception_resnet_v1_arg_scope

from multitask_model import losses, read_and_decode, build_model
import os
import time
slim = tf.contrib.slim

#================ DATASET INFORMATION ======================
data_file = './data/train.tfrecords'

# pre-trained checkpoint
pretrained_checkpoint = './model/model-20170512-110547.ckpt-250000'

#State where your log file is at. If it doesn't exist, create it.
log_dir = './log/model_0_1/'

checkpoint_prefix = log_dir + '/model_iters'
final_checkpoint_file = checkpoint_prefix + '_final.ckpt'

#input for nn.
image_size = 200

#State the number of classes to predict:
num_races = 5
num_samples = 57234

#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 1

#State your batch size
batch_size = 32

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.01
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

num_batches_per_epoch = int(num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

lr = 0.001
decay_rate=0.1
decay_per=40 #epoch
num_iter = 57234 / batch_size



def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    # with tf.Graph().as_default() as graph:

    tf.reset_default_graph()
    # graph = tf.get_default_graph()

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    # create the dataset and load one batch
    images, genders, races = read_and_decode(data_file, [image_size, image_size], batch_size=batch_size)

    # build the multitask model
    logits_gender, logits_race, end_points, variables_to_restore = build_model(images)

    loss_genders = losses(logits_gender, slim.one_hot_encoding(genders, 2))
    loss_races = losses(logits_race, slim.one_hot_encoding(races, num_races))

    # Create the train_op.
    loss = loss_genders + loss_races
    # loss = loss_genders
    end_points['Predictions/gender'] = tf.nn.softmax(logits_gender, name='Predictions/gender')
    end_points['Predictions/race'] = tf.nn.softmax(logits_race, name='Predictions/race')
    predictions1 = tf.argmax(end_points['Predictions/gender'], 1)
    predictions2 = tf.argmax(end_points['Predictions/race'], 1)

    accuracy1 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions1), genders)))
    accuracy2 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions2), races)))

    global_step = get_or_create_global_step()


    # Define your exponentially decaying learning rate
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    # Now we can define the optimizer that takes on the learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = slim.learning.create_train_op(loss, optimizer)


    # Now finally create all the summaries you need to monitor and group them into one summary op.
    tf.summary.scalar('losses/Total_Loss', loss)
    tf.summary.scalar('accuracy_gender', accuracy1)
    tf.summary.scalar('accuracy_race', accuracy2)
    tf.summary.scalar('learning_rate', lr)
    summary_op = tf.summary.merge_all()

    def train_step(sess, train_op, global_step):
        '''
        Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
        '''
        # Check the time for each sess run
        start_time = time.time()
        total_loss, global_step_count, summary = sess.run([train_op, global_step, summary_op])
        time_elapsed = time.time() - start_time

        # Run the logging to print some results
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

        return total_loss, global_step_count, summary

    saver = tf.train.Saver()

    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        input_saver = tf.train.Saver(variables_to_restore)

        sess.run(init_op)
        input_saver.restore(sess, pretrained_checkpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        avg_loss, acc = 0, 0
        for epoch in range(num_epochs):
            logging.info('Epoch %s/%s', epoch+1, num_epochs)

            for step in range(200+1): # num_steps_per_epoch
                l, step_count, curr_summary = train_step(sess, train_op, global_step)
                avg_loss += l / num_iter
                print("Step%03d loss: %f" % (step + 1, l))
                writer.add_summary(curr_summary, epoch * num_batches_per_epoch + step)

                # print more detailed loss and accuracy report every n iterations
                if step % 100 == 0:
                    learning_rate_value, accuracy_value1, accuracy_value2 = sess.run([lr, accuracy1, accuracy2])
                    l_gender, l_race = sess.run([loss_genders, loss_races])

                    print ('loss for gender: ', l_gender)
                    print ('loss for race: ', l_race)

                    # accuracy_value1, accuracy_value2 = sess.run([accuracy1, accuracy2])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current gender Accuracy: %s', accuracy_value1)
                    logging.info('Current race Accuracy: %s', accuracy_value2)
                    logits_value1, predictions_value1, labels_value1 = sess.run(
                        [logits_gender, predictions1, genders])
                    print('logits gender: \n', logits_value1)
                    print('predictions gender: \n', predictions_value1)
                    print('Labels gender:\n', labels_value1)

                    logits_value2, predictions_value2, labels_value2 = sess.run(
                        [logits_race, predictions2, races])
                    print('logits race: \n', logits_value2)
                    print('predictions race: \n', predictions_value2)
                    print('Labels race:\n', labels_value2)

                    saver.save(sess, checkpoint_prefix, global_step=step_count)


            print("Epoch%03d avg_loss: %f" % (epoch, avg_loss))

        # We log the final training loss and accuracy
        logging.info('Final Loss: %s', avg_loss)

        #     # Once all the training has been done, save the log files and checkpoint model
        logging.info('Finished training! Saving model to disk now.')
        # saver.save(sess, "./log/model-gender.ckpt")
        #     sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
        # # Run the managed session
        coord.request_stop()
        coord.join(threads)

        saver.save(sess, final_checkpoint_file)
        sess.close()


if __name__ == '__main__':
    run()