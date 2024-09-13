import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
import time
import random
import datetime
import cv2
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_UP, ROUND_DOWN
from datetime import datetime
from my_utils.my_utils import im2uint8
from my_utils.my_utils import resnet_block
from my_utils.my_utils import lap_sharp
# from my_utils.my_utils import perceptual_loss
# from my_utils.my_utils import total_variation_loss
from my_utils.BasicConvLSTMCell import BasicConvLSTMCell
import tf_slim as slim


class DEBLUR(object):
    def __init__(self, args):
        self.data_queue = None
        self.loss_total = None
        self.all_vars = None
        self.g_vars = None
        self.lstm_vars = None
        self.global_step = None
        self.lr = None
        self.sess = None
        self.saver = None
        self.args = args
        self.lstm_num_features = args.lstm_num_features
        self.post_sharp = args.post_sharp
        self.cont = args.cont
        self.num_iters = args.num_iters
        self.sharpness = args.sharpness
        self.step = args.num_steps
        self.n_levels = args.num_scales
        self.scale = args.scale_factor
        self.chnls = 3  # if self.args.model == 'color' else 1  # input / output channels
        # if args.phase == 'train':
        self.crop_size = args.train_crop_size
        # read the text file into a list of lines, the lines contain ground truth on the left and input on the right separated by a space
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        # map string split function on to each element of the previous list which separate the strings into a list of 2 strings, ground truth [0] and input [1]
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = int(Decimal(len(self.data_list) / self.batch_size).quantize(0, ROUND_HALF_UP))
        self.max_steps = self.epoch * self.data_size
        self.learning_rate = args.learning_rate
        self.gpu = self.args.gpu_id

    def input_producer(self, batch_size=10):
        def preprocessing(imgs):
            # turn the image pixel values into float and normalize them to a 0-1 range
            imgs = [tf.compat.v1.cast(img, tf.compat.v1.float32) / 255.0 for img in imgs]
            # Do if model is not color (deprecated)
            '''
            if self.args.model != 'color':
                imgs = [tf.compat.v1.image.rgb_to_grayscale(img) for img in imgs]
            '''
            img_crop = tf.compat.v1.unstack(tf.compat.v1.random_crop(tf.compat.v1.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chnls]), axis=0)
            return img_crop

        def read_data():
            # print(self.data_queue[0])
            # read input image files
            img_a = tf.compat.v1.image.decode_image(tf.compat.v1.read_file(tf.compat.v1.string_join(['./training_set/', self.data_queue[0]])), channels=3)
            # print(self.data_queue[1])
            # read ground truth image files
            img_b = tf.compat.v1.image.decode_image(tf.compat.v1.read_file(tf.compat.v1.string_join(['./training_set/', self.data_queue[1]])), channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        with tf.compat.v1.variable_scope('input'):
            list_all = tf.compat.v1.convert_to_tensor(self.data_list, dtype=tf.compat.v1.string)
            # All rows in the first column is target/ground truth because it is that way in the data text file
            gt_list = list_all[:, 0]
            # All rows in the second column is input blurry image because it is that way in the data text file
            in_list = list_all[:, 1]
            # now input should be [0] and ground truth is [1]
            self.data_queue = tf.compat.v1.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.compat.v1.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)
        return batch_in, batch_gt

    def generator(self, inputs, reuse=False, scope='g_net'):
        # number of images in a batch?, height, width, channels
        n, h, w, c = inputs.get_shape().as_list()
        # lstm place holder
        if self.args.model == 'lstm':
            with tf.compat.v1.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([int(Decimal(h / 4).quantize(0, ROUND_HALF_UP)), int(Decimal(w / 4).quantize(0, ROUND_HALF_UP))], [3, 3], self.lstm_num_features)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.compat.v1.float32)
        # x_unwrap is list of output from all the scales
        x_unwrap = []
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.compat.v1.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=keras.initializers.glorot_uniform,
                                biases_initializer=keras.initializers.constant(0.0)):  # tf.compat.v1.constant_initializer(0.0)):
                # input predecessors?
                inp_pred = inputs
                # build model input at different scales from the smallest scale up
                for i in range(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    # calculate new height
                    hi = int(Decimal(h * scale).quantize(0, ROUND_HALF_UP))
                    hi = hi - (hi - (int(Decimal(hi / 4).quantize(0, ROUND_DOWN)) * 4))
                    # calculate new width
                    wi = int(Decimal(w * scale).quantize(0, ROUND_HALF_UP))
                    wi = wi - (wi - (int(Decimal(wi / 4).quantize(0, ROUND_DOWN)) * 4))
                    # preprocessing - resize
                    inp_blur = tf.compat.v1.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.compat.v1.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.compat.v1.image.resize_images(rnn_state, [int(Decimal(hi / 4).quantize(0, ROUND_HALF_UP)), int(Decimal(wi / 4).quantize(0, ROUND_HALF_UP))], method=0)
                    # encoder
                    # inp_all_dropout = tf.nn.dropout(inp_all, rate=0.1, seed=1337)
                    conv1_1 = slim.conv2d(inp_all, self.lstm_num_features // 4, [5, 5], scope='enc1_1')
                    conv1_2 = resnet_block(conv1_1, self.lstm_num_features // 4, 5, scope='enc1_2')
                    conv1_3 = resnet_block(conv1_2, self.lstm_num_features // 4, 5, scope='enc1_3')
                    conv1_4 = resnet_block(conv1_3, self.lstm_num_features // 4, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, self.lstm_num_features // 2, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = resnet_block(conv2_1, self.lstm_num_features // 2, 5, scope='enc2_2')
                    conv2_3 = resnet_block(conv2_2, self.lstm_num_features // 2, 5, scope='enc2_3')
                    conv2_4 = resnet_block(conv2_3, self.lstm_num_features // 2, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, self.lstm_num_features, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = resnet_block(conv3_1, self.lstm_num_features, 5, scope='enc3_2')
                    conv3_3 = resnet_block(conv3_2, self.lstm_num_features, 5, scope='enc3_3')
                    conv3_4 = resnet_block(conv3_3, self.lstm_num_features, 5, scope='enc3_4')
                    # lstm layer
                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4
                    # decoder
                    deconv3_3 = resnet_block(deconv3_4, self.lstm_num_features, 5, scope='dec3_3')
                    deconv3_2 = resnet_block(deconv3_3, self.lstm_num_features, 5, scope='dec3_2')
                    deconv3_1 = resnet_block(deconv3_2, self.lstm_num_features, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, self.lstm_num_features // 2, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = resnet_block(cat2, self.lstm_num_features // 2, 5, scope='dec2_3')
                    deconv2_2 = resnet_block(deconv2_3, self.lstm_num_features // 2, 5, scope='dec2_2')
                    deconv2_1 = resnet_block(deconv2_2, self.lstm_num_features // 2, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, self.lstm_num_features // 4, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = resnet_block(cat1, self.lstm_num_features // 4, 5, scope='dec1_3')
                    deconv1_2 = resnet_block(deconv1_3, self.lstm_num_features // 4, 5, scope='dec1_2')
                    deconv1_1 = resnet_block(deconv1_2, self.lstm_num_features // 4, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chnls, [5, 5], activation_fn=None, scope='dec1_0')
                    # Add to x_unwrap output from lower smaller scales as input predecessor, last one is actually final fullscale output
                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.compat.v1.get_variable_scope().reuse_variables()
            return x_unwrap

    def build_model(self):
        # images here have been put into batches
        img_in, img_gt = self.input_producer(self.batch_size)
        # print(img_in.get_shape())
        # print(img_in)
        tf.compat.v1.summary.image('img_in', im2uint8(img_in))
        tf.compat.v1.summary.image('img_gt', im2uint8(img_gt))
        # print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())
        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.compat.v1.image.resize_images(img_gt, [hi, wi], method=0)
            # euclidean loss
            loss = tf.compat.v1.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            # perceptual loss
            # loss = self.perceptual_loss(gt_i, x_unwrap[i])
            self.loss_total += loss
            tf.compat.v1.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.compat.v1.summary.scalar('loss_' + str(i), loss)
        # losses
        tf.compat.v1.summary.scalar('loss_total', self.loss_total)
        # training vars
        all_vars = tf.compat.v1.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        # for var in all_vars:
        #   print(var.name)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(' [*] Reading checkpoints...', end='\r')
        model_name = "deblur.model"
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(' [*] Reading intermediate checkpoints... Success\t\t\t\t\t')
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(' [*] Reading updated checkpoints... Success\t\t\t\t\t')
            return ckpt_iter
        else:
            print(' [*] Reading checkpoints... ERROR\t\t\t\t\t')
            return False

    def train(self):
        def get_optimizer(loss, current_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)  # keras.optimizers.Adam(self.learning_rate)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if 'LSTM' not in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.compat.v1.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                # train_op.iterations = global_step
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=current_step)
            else:
                # train_op.iterations = global_step
                train_op = train_op.minimize(loss, current_step, var_list)
            return train_op

        if self.cont:
            global_step = tf.compat.v1.Variable(initial_value=self.step, dtype=tf.compat.v1.int32, trainable=False)
            self.global_step = global_step
            # build model
            self.build_model()
            # learning rate decay
            self.lr = tf.compat.v1.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0, power=0.3)
            tf.compat.v1.summary.scalar('learning_rate', self.lr)
            # training operators
            if self.args.model == 'lstm':
                train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars, True)
            else:
                train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)
            # session and thread
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            self.sess = sess
            sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.train_dir, step=self.step)
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        else:
            global_step = tf.compat.v1.Variable(initial_value=0, dtype=tf.compat.v1.int32, trainable=False)
            self.global_step = global_step
            # build model
            self.build_model()
            # learning rate decay
            self.lr = tf.compat.v1.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0, power=0.3)
            tf.compat.v1.summary.scalar('learning_rate', self.lr)
            # training operators
            if self.args.model == 'lstm':
                train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars, True)
            else:
                train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)
            # session and thread
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            self.sess = sess
            sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        # training summary
        summary_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)
        # print(self.max_steps)
        # print(self.gpu)

        for step in range(sess.run(global_step), self.max_steps + 1):
            start_time = time.time()
            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])
            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'
            # printing for progress report
            # if step % 5 == 0:
            num_examples_per_step = self.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            epoch = int(Decimal(step / self.data_size).quantize(0, ROUND_UP))

            format_str = '%s - Epoch %d: step %d/%d, loss = %.5f (%.1f data/s; %.3f s/bch)\t\t\t\t\t\t\t\t'
            print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, (step + 1), self.max_steps, loss_total_val, examples_per_sec, sec_per_batch), end='\r')

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            chkpnt = self.data_size*5
            # Save the model checkpoint periodically.
            if step % chkpnt == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)
        print()

    def test(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_names = sorted(os.listdir(input_path))
        num_img = len(image_names)
        H, W = height, width
        inp_chnls = 3  # if self.args.model == 'color' else 1
        self.batch_size = 1  # if self.args.model == 'color' else 3
        inputs = tf.compat.v1.placeholder(shape=[self.batch_size, H, W, inp_chnls], dtype=tf.compat.v1.float32)
        outputs = self.generator(inputs, reuse=False)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        self.saver = tf.compat.v1.train.Saver()
        self.load(sess, self.train_dir, step=self.step)

        for n in range(num_img):
            image_name = image_names[n]
            blur = cv2.imread(os.path.join(input_path, image_name))
            # Original code without iterative processing
            '''
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)

                # blur = skimage.transform.resize(blur, [new_h, new_w], 'bicubic')
                resized_image = PIL.Image.fromarray(blur).resize((new_h, new_w), resample=PIL.Image.Resampling.BICUBIC)
                blur = np.array(resized_image)

                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                #blurPad = np.transpose(blurPad, (3, 1, 2, 0))
            '''
            # Enable iterative processing
            deblur = blur
            h, w, c = blur.shape
            new_h, new_w = (h, w)
            resize = False
            rot = False
            output_img = blur
            start = time.time()
            # Start processing loop
            for i in range(self.num_iters):
                # Pre-process
                # make sure the width is larger than the height
                if h > w:
                    output_img = np.transpose(output_img, [1, 0, 2])
                    rot = True
                h = int(output_img.shape[0])
                w = int(output_img.shape[1])
                if h > H or w > W:
                    scale = min(1.0 * H / h, 1.0 * W / w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    # blur = skimage.transform.resize(blur, [new_h, new_w], 'bicubic')
                    resized_image = PIL.Image.fromarray(output_img).resize((new_w, new_h), resample=PIL.Image.Resampling.BICUBIC)
                    output_img = np.array(resized_image)
                    resize = True
                    blur_pad = np.pad(output_img, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
                else:
                    blur_pad = np.pad(output_img, ((0, H - h), (0, W - w), (0, 0)), 'edge')
                blur_pad = np.expand_dims(blur_pad, 0)
                # Process
                deblur = sess.run(outputs, feed_dict={inputs: blur_pad / 255.0})
                # Post-process
                if i < (self.num_iters - 1):
                    output_img = deblur[-1]
                    output_img = np.clip(output_img[0, :, :, :], 0.0, 1.0) * 255.0
                    # if resized
                    if resize:
                        output_img = output_img[:new_h, :new_w, :]
                        resized_image = PIL.Image.fromarray(output_img).resize((h, w), resample=PIL.Image.Resampling.BICUBIC)
                        output_img = np.array(resized_image)
                    else:
                        output_img = output_img[:h, :w, :]
                    # if rotated
                    if rot:
                        output_img = np.transpose(output_img, [1, 0, 2])
                    # Apply sharpening on non-final output
                    # if i == (self.num_iters - 2):
                        # output_img = lap_sharp(output_img, self.sharpness)
                        # cv2.imwrite(os.path.join("./testing_dir\\input", imgName), (loop_img[:h, :w, :]).astype(np.uint8))
                    # Save
                    img_save = output_img.astype(np.uint8)
                    cv2.imwrite(os.path.join(output_path, image_name.split('.pn')[0] + '(' + str(i + 1) + ').png'), img_save)
            # End processing loop
            duration = time.time() - start

            print('Saving results: %s ... %.3fs | %d/%d\t\t\t\t\t' % (os.path.join(output_path, image_name), duration, (n + 1), num_img), end='\r')
            res_ = deblur[-1]
            # Do if model is not color (deprecated)
            '''
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            '''
            res = np.clip(res_[0, :, :, :], 0.0, 1.0) * 255.0
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                resized_image = PIL.Image.fromarray(res).resize((h, w), resample=PIL.Image.Resampling.BICUBIC)
                res = np.array(resized_image)
            else:
                res = res[:h, :w, :]
            # if rotated
            if rot:
                res = np.transpose(res, [1, 0, 2])
            if self.post_sharp:
                res = lap_sharp(res, self.sharpness)
            res = res.astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, image_name), res)
        print()

    def deblur(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        inp_chnls = 3  # if self.args.model == 'color' else 1
        self.batch_size = 1  # if self.args.model == 'color' else 3
        inputs = tf.compat.v1.placeholder(shape=[self.batch_size, height, width, inp_chnls],
                                          dtype=tf.compat.v1.float32)
        outputs = self.generator(inputs, reuse=False)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        self.saver = tf.compat.v1.train.Saver()
        self.load(sess, self.train_dir, step=self.step)

        image_name = input_path.split('\\')[-1]
        blur = cv2.imread(input_path)

        output_img = blur
        # Start processing loop
        # Pre-process
        blur_pad = np.pad(output_img, ((0, 0), (0, 0), (0, 0)), 'edge')
        blur_pad = np.expand_dims(blur_pad, 0)
        # Process
        deblur = sess.run(outputs, feed_dict={inputs: blur_pad / 255.0})
        res_ = deblur[-1]
        # Do if model is not color (deprecated)
        '''
        if self.args.model != 'color':
            res = np.transpose(res, (3, 1, 2, 0))
        res = im2uint8(res[0, :, :, :])
        '''
        res = np.clip(res_[0, :, :, :], 0.0, 1.0) * 255.0
        # crop the image into original size
        res = res[:height, :width, :]
        if self.post_sharp:
            res = lap_sharp(res, self.sharpness)
        res = res.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, image_name), res)
