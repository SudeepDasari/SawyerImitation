import tensorflow as tf
import numpy as np


NUM_FRAMES = 60
NUM_JOINTS = 7
VELOCITY_DIM = 13
EE_DIM = 7
IMG_WIDTH = 224
IMG_HEIGHT = 224
COLOR_CHANNELS = 3


def read_tf_record(data_path, d_append='train', shuffle = True, rng = 0.3):
    #gets data path
    # data_path = FLAGS.data_path

    #setup tf session
    # with tf.Session() as sess:
        #feature dictionary definition
    feature = {
        d_append+'/image': tf.FixedLenFeature([IMG_WIDTH * IMG_HEIGHT * NUM_FRAMES * 3], tf.string),
        d_append+'/angle': tf.FixedLenFeature([NUM_JOINTS * NUM_FRAMES], tf.float32),
        d_append+'/velocity': tf.FixedLenFeature([VELOCITY_DIM * NUM_FRAMES], tf.float32),
        d_append+'/endeffector_pos': tf.FixedLenFeature([EE_DIM * NUM_FRAMES], tf.float32)
    }


    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)


    # Convert the image and robot data from string back to the numbers
    image = tf.decode_raw(features[d_append+'/image'], tf.uint8)
    angle = tf.reshape(features[d_append+'/angle'], shape=[NUM_FRAMES, NUM_JOINTS])
    velocity = tf.reshape(features[d_append+'/velocity'], shape=[NUM_FRAMES, VELOCITY_DIM])
    endeffector_pos = tf.reshape(features[d_append+'/endeffector_pos'], shape=[NUM_FRAMES, EE_DIM])

    final_endeffector_pos = tf.reshape(tf.tile(endeffector_pos[-1:, :], [NUM_FRAMES, 1]), shape=[NUM_FRAMES, EE_DIM])

    use_frame = np.zeros(NUM_FRAMES)
    use_frame[0] = 1

    use_frame_queue = tf.train.input_producer(
        [tf.reshape(tf.convert_to_tensor(use_frame, dtype=tf.float32), shape=[NUM_FRAMES, 1])],
        element_shape=[NUM_FRAMES, 1],
        shuffle=False)
    use_frame = use_frame_queue.dequeue()

    # Reshape image data into original video
    image = tf.reshape(image, [NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, COLOR_CHANNELS])

    image_rgb = image[:, :, :, ::-1]
    image_rgb = tf.image.convert_image_dtype(image_rgb, tf.float32)
    image_hsv = tf.image.rgb_to_hsv(image_rgb)
    img_stack = [tf.unstack(i, axis = 2) for i in tf.unstack(image_hsv, axis = 0)]
    stack_mod = [tf.stack([x[0] + tf.random_uniform([1], minval = -rng, maxval = rng),
                           x[1] + tf.random_uniform([1], minval = -rng, maxval = rng),
                           x[2] + tf.random_uniform([1], minval = -rng, maxval = rng)]
                          ,axis = 2) for x in img_stack]

    image_rgb = tf.image.hsv_to_rgb(tf.stack(stack_mod))
    image_rgb = tf.image.convert_image_dtype(image_rgb, tf.uint8, saturate=True)
    image = image_rgb[:, :, :, ::-1]

    # Creates batches by randomly shuffling tensors. each training example is (image,velocity) pair
    if shuffle:
        images, angles, velocities, endeffector_poses, use_frames, final_endeffector_poses = \
            tf.train.shuffle_batch([image, angle, velocity, endeffector_pos, use_frame, final_endeffector_pos],
                                   batch_size=30, capacity=3000, num_threads=30,
                                   min_after_dequeue=900, enqueue_many=True)
    else:
        images, angles, velocities, endeffector_poses, use_frames, final_endeffector_poses = \
            tf.train.batch([image, angle, velocity, endeffector_pos, use_frame, final_endeffector_pos],
                                   batch_size=30, capacity=3000, num_threads=30, enqueue_many=True)
    joint_velocity_dim = velocities[:, :7]
    ee_velocity_dim = velocities[:, 7:]
    return images, angles, joint_velocity_dim, endeffector_poses, ee_velocity_dim, use_frames, final_endeffector_poses


def main():
    from tensorflow.python.platform import flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string('data_path', './', 'path to tfrecords file')

    import cv2
    import matplotlib.pyplot as plt
        # # Initialize all global and local variables
    images, angles, joint_velocities, endeffector_poses, ee_vel, use_frames, final_eeps = read_tf_record(FLAGS.data_path, shuffle=False)

    with tf.Session(config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )) as sess:



        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #
        for batch_index in range(1):

            img, j_vel, ef, ang, uf, fef, evel = sess.run([images, joint_velocities, endeffector_poses, angles, use_frames, final_eeps, ee_vel])


            print 'batch_index', batch_index
            print 'vel', j_vel.shape
            print 'ef_vel', evel.shape
            print 'ef', ef.shape

            for i in img:
                cv2.imshow('img', i)
                cv2.waitKey(1000)

            # plt.plot(vel[:, 0])
            # plt.figure()
            # plt.plot(ang[:, 0])
            # plt.show()
    #     # Stop the threads
        coord.request_stop()
    #
    #     # Wait for threads to stop
    #     coord.join(threads)
    sess.close()



if __name__ == '__main__':
    main()
