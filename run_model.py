import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import argparse
# import models.model_gray as model
# import models.model_color as model
import srn_model.srn_model as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--lstm_num_features', type=int, default=128, help='number of features for lstm layer')
    parser.add_argument('--train_crop_size', type=int, default=256, help='size of the random crop used in training')
    parser.add_argument('--post_sharp', action=argparse.BooleanOptionalAction, default=False, help='apply post process sharpening?')
    parser.add_argument('--cont', action=argparse.BooleanOptionalAction, default=False, help='continue training?')
    parser.add_argument('--num_iters', type=int, default=1, help='how many times an image will be processed')
    parser.add_argument('--sharpness', type=float, default=0.0430237, help='how much sharpness to apply in processing')
    parser.add_argument('--num_steps', type=int, default=523000, help='choose test model by number of steps')
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--scale_factor', type=float, default=0.5, help='scaling factor, determines the resolution of the image in the next scale and the on after that ...')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default='./datalist.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720, help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280, help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='./testing_set', help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_dir/results', help='output path for testing images')
    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()
    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.input_path, args.output_path)
    elif args.phase == 'train':
        deblur.train()
    elif args.phase == 'run':
        deblur.deblur(args.height, args.width, args.input_path, args.output_path)
    else:
        print('phase should be set to either test or train')

if __name__ == '__main__':
    tf.compat.v1.app.run()