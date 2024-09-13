import skimage.metrics as metrics
import skimage.io as io
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

histograms_path = "./histograms"
gt_path = "./gt"
results_1 = "./blur"
results_2 = "./results"

images = os.listdir(gt_path)

#gt_files = os.listdir("./gt")
#print(len(gt_files))

psnr_results_1 = []
ssim_results_1 = []
psnr_results_2 = []
ssim_results_2 = []
num_files = len(images)
for i in range(20):
    image = images[i]
    print('Calculating metrics for image ' + str(i + 1) + '/' + str(num_files) + '  ', end='\r')
    result_1 = io.imread(os.path.join(results_1, image))
    result_2 = io.imread(os.path.join(results_2, image))
    image_true = io.imread(os.path.join(gt_path, image))

    psnr_1 = metrics.peak_signal_noise_ratio(image_true, result_1)
    psnr_2 = metrics.peak_signal_noise_ratio(image_true, result_2)
    psnr_results_1.append(psnr_1)
    psnr_results_2.append(psnr_2)

    ssim_1 = metrics.structural_similarity(im1=image_true, im2=result_1, channel_axis=2)
    ssim_results_1.append(ssim_1)
    ssim_2 = metrics.structural_similarity(im1=image_true, im2=result_2, channel_axis=2)
    ssim_results_2.append(ssim_2)

print()
'''
if num_files % 2 == 0:
    psnr1_q2 = (psnr_results_1[num_files // 2] + psnr_results_1[(num_files // 2) + 1]) / 2
    psnr2_q2 = (psnr_results_2[num_files // 2] + psnr_results_2[(num_files // 2) + 1]) / 2

    psnr1_q1 = (psnr_results_1[(num_files // 2) // 2] + psnr_results_1[((num_files // 2) // 2) + 1]) / 2
    psnr1_q3 = (psnr_results_1[(num_files // 2) + ((num_files // 2) // 2)] + psnr_results_1[((num_files // 2) + ((num_files // 2) // 2)) + 1]) / 2

    psnr2_q1 = (psnr_results_2[(num_files // 2) // 2] + psnr_results_2[((num_files // 2) // 2) + 1]) / 2
    psnr2_q3 = (psnr_results_2[(num_files // 2) + ((num_files // 2) // 2)] + psnr_results_2[((num_files // 2) + ((num_files // 2) // 2)) + 1]) / 2
else:
    psnr1_q2 = psnr_results_1[math.ceil(num_files / 2)]
    psnr2_q2 = psnr_results_2[math.ceil(num_files / 2)]

    psnr1_q1 = psnr_results_1[math.ceil(math.floor(num_files / 2) / 2)]
    psnr1_q3 = psnr_results_1[math.ceil(num_files / 2) + math.ceil(math.floor(num_files / 2) / 2)]

    psnr2_q1 = psnr_results_2[math.ceil(math.floor(num_files / 2) / 2)]
    psnr2_q3 = psnr_results_2[math.ceil(num_files / 2) + math.ceil(math.floor(num_files / 2) / 2)]
'''

psnr_avg_1 = sum(psnr_results_1) / len(psnr_results_1)
print("PSNR Average Before:" + str(psnr_avg_1))
psnr_avg_2 = sum(psnr_results_2) / len(psnr_results_2)
print("PSNR Average After:" + str(psnr_avg_2))

ssim_avg_1 = sum(ssim_results_1) / len(ssim_results_1)
print("SSIM Average Before:" + str(ssim_avg_1))
ssim_avg_2 = sum(ssim_results_2) / len(ssim_results_2)
print("SSIM Average After:" + str(ssim_avg_2))
'''
plt.boxplot(psnr_results_1)
plt.boxplot(psnr_results_2)
plt.savefig(os.path.join(histograms_path, 'hist_summary.png'))
'''
'''
file = open(os.path.join(histograms_path, 'summary.txt'), 'w')
file.write('PSNR Average Before : ' + f'{psnr_results_1[i]:.4f}' + ' | ' + f'{psnr_results_2[i]:.4f}\n')
file.write('SSIM Before | After : ' + f'{ssim_results_1[i]:.4f}' + ' | ' + f'{ssim_results_2[i]:.4f}\n')
'''

output_alpha_1 = 0.25
output_alpha_2 = 0.5
output_color_r = 'red'
output_color_g = 'green'
output_color_b = 'blue'
output_color_total = 'gold'

channels_alpha_1 = 0.25
channels_alpha_2 = 0.5
channel_color_1 = 'black'
channel_color_2 = 'orangered'
channel_color_3 = 'turquoise'


for i in range(20):
    print('Processing file ' + str(i + 1) + '/' + str(num_files) + '\t\t\t\t', end='\r')
    if psnr_results_2[i] > psnr_avg_2 and psnr_results_2[i] > psnr_results_1[i]:
        image = images[i]
        path = os.path.join(histograms_path, image.split('.pn')[0])
        if not os.path.exists(path):
            os.makedirs(path)
        image_gt = cv2.imread(os.path.join(gt_path, image))
        image_1 = cv2.imread(os.path.join(results_1, image))
        image_2 = cv2.imread(os.path.join(results_2, image))

        _image_gt_r = cv2.calcHist([image_gt], [2], None, [256], [0, 256])
        image_gt_r = np.array([int(count) for counts in _image_gt_r for count in counts])
        _image_1_r = cv2.calcHist([image_1], [2], None, [256], [0, 256])
        image_1_r = np.array([int(count) for counts in _image_1_r for count in counts])
        _image_2_r = cv2.calcHist([image_2], [2], None, [256], [0, 256])
        image_2_r = np.array([int(count) for counts in _image_2_r for count in counts])

        _image_gt_g = cv2.calcHist([image_gt], [1], None, [256], [0, 256])
        image_gt_g = np.array([int(count) for counts in _image_gt_g for count in counts])
        _image_1_g = cv2.calcHist([image_1], [1], None, [256], [0, 256])
        image_1_g = np.array([int(count) for counts in _image_1_g for count in counts])
        _image_2_g = cv2.calcHist([image_2], [1], None, [256], [0, 256])
        image_2_g = np.array([int(count) for counts in _image_2_g for count in counts])

        _image_gt_b = cv2.calcHist([image_gt], [0], None, [256], [0, 256])
        image_gt_b = np.array([int(count) for counts in _image_gt_b for count in counts])
        _image_1_b = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        image_1_b = np.array([int(count) for counts in _image_1_b for count in counts])
        _image_2_b = cv2.calcHist([image_2], [0], None, [256], [0, 256])
        image_2_b = np.array([int(count) for counts in _image_2_b for count in counts])

        image_gt_total = np.add(image_gt_r, np.add(image_gt_g, image_gt_b))
        image_1_total = np.add(image_1_r, np.add(image_1_g, image_1_b))
        image_2_total = np.add(image_2_r, np.add(image_2_g, image_2_b))
        x=np.array(range(0, 256))

        # Ground truth
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_output_target.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_total, color=output_color_total, alpha = output_alpha_1)
        plt.bar(x, image_gt_r, color=output_color_r, alpha = output_alpha_2)
        plt.bar(x, image_gt_g, color=output_color_g, alpha = output_alpha_2)
        plt.bar(x, image_gt_b,  color=output_color_b, alpha = output_alpha_2)
        plt.legend(['Total', 'Red', 'Green', 'Blue'])
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Ground Truth')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_output_target.png'))
        plt.clf()

        # Before
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_output_before.png\t\t\t\t', end='\r')
        plt.bar(x, image_1_total, color=output_color_total, alpha=output_alpha_1)
        plt.bar(x, image_1_r, color=output_color_r, alpha=output_alpha_2)
        plt.bar(x, image_1_g, color=output_color_g, alpha=output_alpha_2)
        plt.bar(x, image_1_b, color=output_color_b, alpha=output_alpha_2)
        plt.legend(['Total', 'Red', 'Green', 'Blue'])
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Before')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_output_before.png'))
        plt.clf()

        # After
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_output_post.png\t\t\t\t', end='\r')
        plt.bar(x, image_2_total, color=output_color_total, alpha=output_alpha_1)
        plt.bar(x, image_2_r, color=output_color_r, alpha=output_alpha_2)
        plt.bar(x, image_2_g, color=output_color_g, alpha=output_alpha_2)
        plt.bar(x, image_2_b, color=output_color_b, alpha=output_alpha_2)
        plt.legend(['Total', 'Red', 'Green', 'Blue'])
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' After')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_output_post.png'))
        plt.clf()

        # Red
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_red.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_r, color=channel_color_1, alpha=channels_alpha_1)
        plt.bar(x, image_1_r, color=channel_color_2, alpha=channels_alpha_2)
        plt.bar(x, image_2_r, color=channel_color_3, alpha=channels_alpha_2)
        plt.legend(['Ground truth', 'Before', 'After'])
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Red')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_red.png'))
        plt.clf()

        # Green
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_green.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_g, color=channel_color_1, alpha=channels_alpha_1)
        plt.bar(x, image_1_g, color=channel_color_2, alpha=channels_alpha_2)
        plt.bar(x, image_2_g, color=channel_color_3, alpha=channels_alpha_2)
        plt.legend(['Ground truth', 'Before', 'After'])
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Green')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_green.png'))
        plt.clf()

        # Blue
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_blue.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_b, color=channel_color_1, alpha=channels_alpha_1)
        plt.bar(x, image_1_b, color=channel_color_2, alpha=channels_alpha_2)
        plt.bar(x, image_2_b, color=channel_color_3, alpha=channels_alpha_2)
        plt.legend(['Ground truth', 'Before', 'After'])
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Blue')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_blue.png'))
        plt.clf()

        # RGB Total
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_rgb_total.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_total, color=channel_color_1, alpha=channels_alpha_1)
        plt.bar(x, image_1_total, color=channel_color_2, alpha=channels_alpha_2)
        plt.bar(x, image_2_total, color=channel_color_3, alpha=channels_alpha_2)
        plt.legend(['Ground truth', 'Before', 'After'])
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Total')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_rgb_total.png'))
        plt.clf()

        # Red Single
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_red_before.png\t\t\t\t', end='\r')
        plt.bar(x, image_1_r, color='red', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Red Before')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_red_before.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_red_post.png\t\t\t\t', end='\r')
        plt.bar(x, image_2_r, color='red', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Red After')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_red_post.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_red_target.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_r, color='red', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Red Target')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_red_target.png'))
        plt.clf()

        # Green Single
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_green_before.png\t\t\t\t', end='\r')
        plt.bar(x, image_1_g, color='green', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Green Before')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_green_before.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_green_post.png\t\t\t\t', end='\r')
        plt.bar(x, image_2_g, color='green', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Green After')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_green_post.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_green_target.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_g, color='green', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Green Target')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_green_target.png'))
        plt.clf()

        # Blue Single
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_blue_before.png\t\t\t\t', end='\r')
        plt.bar(x, image_1_b, color='blue', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Blue Before')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_blue_before.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_blue_post.png\t\t\t\t', end='\r')
        plt.bar(x, image_2_b, color='blue', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Blue After')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_blue_post.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_channel_blue_target.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_b, color='blue', alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 20000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' Blue Target')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_channel_blue_target.png'))
        plt.clf()

        # Total single
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_rgb_total_before.png\t\t\t\t', end='\r')
        plt.bar(x, image_1_total, color=channel_color_1, alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' RGB Total Before')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_rgb_total_before.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_rgb_total_post.png\t\t\t\t', end='\r')
        plt.bar(x, image_2_total, color=channel_color_1, alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' RGB Total After')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_rgb_total_post.png'))
        plt.clf()
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving hist_' + image.split('.pn')[0] + '_rgb_total_target.png\t\t\t\t', end='\r')
        plt.bar(x, image_gt_total, color=channel_color_1, alpha=channels_alpha_2)
        plt.xlim([0, 256])
        plt.ylim([0, 40000])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(image.split('.pn')[0] + ' RGB Total Target')
        plt.savefig(os.path.join(path, 'hist_' + image.split('.pn')[0] + '_rgb_total_target.png'))
        plt.clf()

        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving ' + image.split('.pn')[0] + '_before.png\t\t\t\t', end='\r')
        cv2.imwrite(os.path.join(path, image.split('.pn')[0]) + '_before.png', image_1)
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving ' + image.split('.pn')[0] + '_post.png\t\t\t\t', end='\r')
        cv2.imwrite(os.path.join(path, image.split('.pn')[0]) + '_post.png', image_2)
        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving ' + image.split('.pn')[0] + '_target.png\t\t\t\t', end='\r')
        cv2.imwrite(os.path.join(path, image.split('.pn')[0]) + '_target.png', image_gt)

        print('Processing file ' + str(i + 1) + '/' + str(num_files) + ' saving metrics.txt\t\t\t\t',end='\r')
        file = open(os.path.join(path, 'metrics.txt'), 'w')
        file.write('PSNR Before | After : ' + f'{psnr_results_1[i]:.4f}' + ' | ' + f'{psnr_results_2[i]:.4f}\n')
        file.write('SSIM Before | After : ' + f'{ssim_results_1[i]:.4f}' + ' | ' + f'{ssim_results_2[i]:.4f}\n')

print()
