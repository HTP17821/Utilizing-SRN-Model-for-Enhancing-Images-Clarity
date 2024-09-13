import cv2
import skimage.metrics as metrics
import skimage.io as io
import os
import math
from matplotlib import pyplot as plt
import numpy as np

output_1_path = './867e_1111_results'
output_2_path = './870e_1111_results'
output_1_name = 'Base_model'
output_2_name = 'Modified_model'
gt_path = './gt'
result_path = './output_analysis'
output_1_result_path = os.path.join(result_path, output_1_name)
output_2_result_path = os.path.join(result_path, output_2_name)

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(output_1_result_path):
    os.makedirs(output_1_result_path)
if not os.path.exists(output_2_result_path):
    os.makedirs(output_2_result_path)

images = os.listdir(gt_path)
num_img = len(images)

psnr_results_1 = []
ssim_results_1 = []
psnr_results_2 = []
ssim_results_2 = []
for i in range(num_img):
    image = images[i]
    print('Calculating metrics for image ' + str(i + 1) + '/' + str(num_img) + ' | ', end='')
    output_1 = io.imread(os.path.join(output_1_path, image))
    output_2 = io.imread(os.path.join(output_2_path, image))
    gt = io.imread(os.path.join(gt_path, image))
    psnr_1 = metrics.peak_signal_noise_ratio(gt, output_1)
    psnr_2 = metrics.peak_signal_noise_ratio(gt, output_2)
    psnr_results_1.append(psnr_1)
    psnr_results_2.append(psnr_2)
    ssim_1 = metrics.structural_similarity(im1=gt, im2=output_1, channel_axis=2)
    ssim_2 = metrics.structural_similarity(im1=gt, im2=output_2, channel_axis=2)
    ssim_results_1.append(ssim_1)
    ssim_results_2.append(ssim_2)
    print(f'PSNR 1//2 = {psnr_1:.4f}//{psnr_2:.4f} , SSIM 1//2 = {ssim_1:.4f}//{ssim_2:.4f}\t\t\t\t\t', end='\r')
print()

psnr_mean_1 = sum(psnr_results_1) / num_img
print(f'PSNR Average 1 = {psnr_mean_1:.5f}')
psnr_mean_2 = sum(psnr_results_2) / num_img
print(f'PSNR Average 2 = {psnr_mean_2:.5f}')

ssim_mean_1 = sum(ssim_results_1) / num_img
print(f'SSIM Average 1 = {ssim_mean_1:.5f}')

ssim_mean_2 = sum(ssim_results_2) / num_img
print(f'SSIM Average 2 = {ssim_mean_2:.5f}')

sorted_psnr_results_1 = [psnr for psnr in psnr_results_1]
sorted_psnr_results_1.sort()
sorted_ssim_results_1 = [ssim for ssim in ssim_results_1]
sorted_ssim_results_1.sort()

sorted_psnr_results_2 = [psnr for psnr in psnr_results_2]
sorted_psnr_results_2.sort()
sorted_ssim_results_2 = [ssim for ssim in ssim_results_2]
sorted_ssim_results_2.sort()

print('Calculating quartiles\t\t\t\t\t', end='\r')
if num_img % 2 == 0:
    psnr_q2_1 = (sorted_psnr_results_1[num_img // 2] + sorted_psnr_results_1[(num_img // 2) + 1]) / 2
    psnr_q1_1 = (sorted_psnr_results_1[(num_img // 2) // 2] + sorted_psnr_results_1[((num_img // 2) // 2) + 1]) / 2
    psnr_q3_1 = (sorted_psnr_results_1[(num_img // 2) + ((num_img // 2) // 2)] + sorted_psnr_results_1[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    psnr_iqr_1 = psnr_q3_1 - psnr_q1_1
    psnr_lower_1 = psnr_q1_1 - 1.5 * psnr_iqr_1
    psnr_upper_1 = psnr_q3_1 + 1.5 * psnr_iqr_1
    psnr_max_1 = sorted_psnr_results_1[-1]
    psnr_min_1 = sorted_psnr_results_1[0]

    psnr_q2_2 = (sorted_psnr_results_2[num_img // 2] + sorted_psnr_results_2[(num_img // 2) + 1]) / 2
    psnr_q1_2 = (sorted_psnr_results_2[(num_img // 2) // 2] + sorted_psnr_results_2[((num_img // 2) // 2) + 1]) / 2
    psnr_q3_2 = (sorted_psnr_results_2[(num_img // 2) + ((num_img // 2) // 2)] + sorted_psnr_results_2[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    psnr_iqr_2 = psnr_q3_2 - psnr_q1_2
    psnr_lower_2 = psnr_q1_2 - 1.5 * psnr_iqr_2
    psnr_upper_2 = psnr_q3_2 + 1.5 * psnr_iqr_2
    psnr_max_2 = sorted_psnr_results_2[-1]
    psnr_min_2 = sorted_psnr_results_2[0]

    ssim_q2_1 = (sorted_ssim_results_1[num_img // 2] + sorted_ssim_results_1[(num_img // 2) + 1]) / 2
    ssim_q1_1 = (sorted_ssim_results_1[(num_img // 2) // 2] + sorted_ssim_results_1[((num_img // 2) // 2) + 1]) / 2
    ssim_q3_1 = (sorted_ssim_results_1[(num_img // 2) + ((num_img // 2) // 2)] + sorted_ssim_results_1[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    ssim_iqr_1 = ssim_q3_1 - ssim_q1_1
    ssim_lower_1 = ssim_q1_1 - 1.5 * ssim_iqr_1
    ssim_upper_1 = ssim_q3_1 + 1.5 * ssim_iqr_1
    ssim_max_1 = sorted_ssim_results_1[-1]
    ssim_min_1 = sorted_ssim_results_1[0]

    ssim_q2_2 = (sorted_ssim_results_2[num_img // 2] + sorted_ssim_results_2[(num_img // 2) + 1]) / 2
    ssim_q1_2 = (sorted_ssim_results_2[(num_img // 2) // 2] + sorted_ssim_results_2[((num_img // 2) // 2) + 1]) / 2
    ssim_q3_2 = (sorted_ssim_results_2[(num_img // 2) + ((num_img // 2) // 2)] + sorted_ssim_results_2[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    ssim_iqr_2 = ssim_q3_2 - ssim_q1_2
    ssim_lower_2 = ssim_q1_2 - 1.5 * ssim_iqr_2
    ssim_upper_2 = ssim_q3_1 + 1.5 * ssim_iqr_2
    ssim_max_2 = sorted_ssim_results_2[-1]
    ssim_min_2 = sorted_ssim_results_2[0]
else:
    psnr_q2_1 = sorted_psnr_results_1[math.ceil(num_img / 2)]
    psnr_q1_1 = sorted_psnr_results_1[math.ceil(math.floor(num_img / 2) / 2)]
    psnr_q3_1 = sorted_psnr_results_1[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    psnr_iqr_1 = psnr_q3_1 - psnr_q1_1
    psnr_lower_1 = psnr_q1_1 - 1.5 * psnr_iqr_1
    psnr_upper_1 = psnr_q3_1 + 1.5 * psnr_iqr_1
    psnr_max_1 = sorted_psnr_results_1[-1]
    psnr_min_1 = sorted_psnr_results_1[0]

    psnr_q2_2 = sorted_psnr_results_2[math.ceil(num_img / 2)]
    psnr_q1_2 = sorted_psnr_results_2[math.ceil(math.floor(num_img / 2) / 2)]
    psnr_q3_2 = sorted_psnr_results_2[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    psnr_iqr_2 = psnr_q3_2 - psnr_q1_2
    psnr_lower_2 = psnr_q1_2 - 1.5 * psnr_iqr_2
    psnr_upper_2 = psnr_q3_2 + 1.5 * psnr_iqr_2
    psnr_max_2 = sorted_psnr_results_2[-1]
    psnr_min_2 = sorted_psnr_results_2[0]

    ssim_q2_1 = sorted_ssim_results_1[math.ceil(num_img / 2)]
    ssim_q1_1 = sorted_ssim_results_1[math.ceil(math.floor(num_img / 2) / 2)]
    ssim_q3_1 = sorted_ssim_results_1[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    ssim_iqr_1 = ssim_q3_1 - ssim_q1_1
    ssim_lower_1 = ssim_q1_1 - 1.5 * ssim_iqr_1
    ssim_upper_1 = ssim_q3_1 + 1.5 * ssim_iqr_1
    ssim_max_1 = sorted_ssim_results_1[-1]
    ssim_min_1 = sorted_ssim_results_1[0]

    ssim_q2_2 = sorted_ssim_results_2[math.ceil(num_img / 2)]
    ssim_q1_2 = sorted_ssim_results_2[math.ceil(math.floor(num_img / 2) / 2)]
    ssim_q3_2 = sorted_ssim_results_2[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    ssim_iqr_2 = ssim_q3_2 - ssim_q1_2
    ssim_lower_2 = ssim_q1_2 - 1.5 * ssim_iqr_2
    ssim_upper_2 = ssim_q3_2 + 1.5 * ssim_iqr_2
    ssim_max_2 = sorted_ssim_results_2[-1]
    ssim_min_2 = sorted_ssim_results_2[0]

print('Saving ' + output_1_name + '_quartiles.txt\t\t\t\t\t', end='\r')
file = open(os.path.join(output_1_result_path, output_1_name + '_quartiles.txt'), 'w')
file.write('============== PSNR ==============\n')
file.write('PSNR Mean: ' + f'{psnr_mean_1:.4f}\n')
file.write('PSNR Q2: ' + f'{psnr_q2_1:.4f}\n')
file.write('PSNR Q1: ' + f'{psnr_q1_1:.4f}\n')
file.write('PSNR Q3: ' + f'{psnr_q3_1:.4f}\n')
file.write('PSNR IQR: ' + f'{psnr_iqr_1:.4f}\n')
file.write('PSNR Lower: ' + f'{psnr_lower_1:.4f}\n')
file.write('PSNR Upper: ' + f'{psnr_upper_1:.4f}\n')
file.write('PSNR Min: ' + f'{psnr_min_1:.4f}\n')
file.write('PSNR Max: ' + f'{psnr_max_1:.4f}\n')
file.write('============== SSIM ==============\n')
file.write('SSIM Mean: ' + f'{ssim_mean_1:.4f}\n')
file.write('SSIM Q2: ' + f'{ssim_q2_1:.4f}\n')
file.write('SSIM Q1: ' + f'{ssim_q1_1:.4f}\n')
file.write('SSIM Q3: ' + f'{ssim_q3_1:.4f}\n')
file.write('SSIM IQR: ' + f'{ssim_iqr_1:.4f}\n')
file.write('SSIM Lower: ' + f'{ssim_lower_1:.4f}\n')
file.write('SSIM Upper: ' + f'{ssim_upper_1:.4f}\n')
file.write('SSIM Min: ' + f'{ssim_min_1:.4f}\n')
file.write('SSIM Max: ' + f'{ssim_max_1:.4f}\n')
file.close()

print('Saving ' + output_2_name + '_quartiles.txt\t\t\t\t\t', end='\r')
file = open(os.path.join(output_2_result_path, output_2_name + '_quartiles.txt'), 'w')
file.write('============== PSNR ==============\n')
file.write('PSNR Mean: ' + f'{psnr_mean_2:.4f}\n')
file.write('PSNR Q2: ' + f'{psnr_q2_2:.4f}\n')
file.write('PSNR Q1: ' + f'{psnr_q1_2:.4f}\n')
file.write('PSNR Q3: ' + f'{psnr_q3_2:.4f}\n')
file.write('PSNR IQR: ' + f'{psnr_iqr_2:.4f}\n')
file.write('PSNR Lower: ' + f'{psnr_lower_2:.4f}\n')
file.write('PSNR Upper: ' + f'{psnr_upper_2:.4f}\n')
file.write('PSNR Min: ' + f'{psnr_min_2:.4f}\n')
file.write('PSNR Max: ' + f'{psnr_max_2:.4f}\n')
file.write('============== SSIM ==============\n')
file.write('SSIM Mean: ' + f'{ssim_mean_2:.4f}\n')
file.write('SSIM Q2: ' + f'{ssim_q2_2:.4f}\n')
file.write('SSIM Q1: ' + f'{ssim_q1_2:.4f}\n')
file.write('SSIM Q3: ' + f'{ssim_q3_2:.4f}\n')
file.write('SSIM IQR: ' + f'{ssim_iqr_2:.4f}\n')
file.write('SSIM Lower: ' + f'{ssim_lower_2:.4f}\n')
file.write('SSIM Upper: ' + f'{ssim_upper_2:.4f}\n')
file.write('SSIM Min: ' + f'{ssim_min_2:.4f}\n')
file.write('SSIM Max: ' + f'{ssim_max_2:.4f}\n')
file.close()

print('Saving ' + output_1_name + '_PSNR_box.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_psnr_results_1, meanline=True, showmeans=True)
plt.title(output_1_name + ' results distribution based on PSNR')
plt.ylabel('PSNR')
plt.grid(True)
plt.savefig(os.path.join(output_1_result_path, output_1_name + '_PSNR_box.png'))
plt.clf()

print('Saving ' + output_2_name + '_PSNR_box.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_psnr_results_2, meanline=True, showmeans=True)
plt.title(output_2_name + ' results distribution based on PSNR')
plt.ylabel('PSNR')
plt.grid(True)
plt.savefig(os.path.join(output_2_result_path, output_2_name + '_PSNR_box.png'))
plt.clf()

print('Saving ' + output_1_name + '_SSIM_box.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_ssim_results_1, meanline=True, showmeans=True)
plt.title(output_1_name + ' results distribution based on SSIM')
plt.ylabel('SSIM')
plt.grid(True)
plt.savefig(os.path.join(output_1_result_path, output_1_name + '_SSIM_box.png'))
plt.clf()

print('Saving ' + output_2_name + '_SSIM_box.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_ssim_results_2, meanline=True, showmeans=True)
plt.title(output_2_name + ' results distribution based on SSIM')
plt.ylabel('SSIM')
plt.grid(True)
plt.savefig(os.path.join(output_2_result_path, output_2_name + '_SSIM_box.png'))
plt.clf()

'''
PSNR_path = os.path.join(dataset_result_path, 'Sorted_by_PSNR')
SSIM_path = os.path.join(dataset_result_path, 'Sorted_by_SSIM')

psnr_upper_xcpt_path = os.path.join(PSNR_path, 'upper_xcpt')
psnr_upper_path = os.path.join(PSNR_path, 'upper')
psnr_q2_3_path = os.path.join(PSNR_path, 'q2_3')
psnr_q1_2_path = os.path.join(PSNR_path, 'q1_2')
psnr_lower_path = os.path.join(PSNR_path, 'lower')
psnr_lower_xcpt_path = os.path.join(PSNR_path, 'lower_xcpt')
psnr_class_count = [0, 0, 0, 0, 0, 0]

ssim_upper_xcpt_path = os.path.join(SSIM_path, 'upper_xcpt')
ssim_upper_path = os.path.join(SSIM_path, 'upper')
ssim_q2_3_path = os.path.join(SSIM_path, 'q2_3')
ssim_q1_2_path = os.path.join(SSIM_path, 'q1_2')
ssim_lower_path = os.path.join(SSIM_path, 'lower')
ssim_lower_xcpt_path = os.path.join(SSIM_path, 'lower_xcpt')
ssim_class_count = [0, 0, 0, 0, 0, 0]

if not os.path.exists(psnr_upper_xcpt_path):
    os.makedirs(psnr_upper_xcpt_path)
if not os.path.exists(psnr_upper_path):
    os.makedirs(psnr_upper_path)
if not os.path.exists(psnr_q2_3_path):
    os.makedirs(psnr_q2_3_path)
if not os.path.exists(psnr_q1_2_path):
    os.makedirs(psnr_q1_2_path)
if not os.path.exists(psnr_lower_path):
    os.makedirs(psnr_lower_path)
if not os.path.exists(psnr_lower_xcpt_path):
    os.makedirs(psnr_lower_xcpt_path)

if not os.path.exists(ssim_upper_xcpt_path):
    os.makedirs(ssim_upper_xcpt_path)
if not os.path.exists(ssim_upper_path):
    os.makedirs(ssim_upper_path)
if not os.path.exists(ssim_q2_3_path):
    os.makedirs(ssim_q2_3_path)
if not os.path.exists(ssim_q1_2_path):
    os.makedirs(ssim_q1_2_path)
if not os.path.exists(ssim_lower_path):
    os.makedirs(ssim_lower_path)
if not os.path.exists(ssim_lower_xcpt_path):
    os.makedirs(ssim_lower_xcpt_path)

for i in range(num_img):
    print('Sorting images ' + str(i + 1) + '/' + str(num_img) + '\t\t\t\t\t', end='\r')
    image_name = images[i]
    image = cv2.imread(os.path.join(blur_path, image_name), flags=cv2.IMREAD_UNCHANGED)
    psnr = psnr_results[i]
    ssim = ssim_results[i]

    if psnr_upper < psnr:
        psnr_class_count[5] += 1
        cv2.imwrite(os.path.join(psnr_upper_xcpt_path, image_name), image)
    elif psnr_q3 < psnr < psnr_upper:
        psnr_class_count[4] += 1
        cv2.imwrite(os.path.join(psnr_upper_path, image_name), image)
    elif psnr_q2 < psnr < psnr_q3:
        psnr_class_count[3] += 1
        cv2.imwrite(os.path.join(psnr_q2_3_path, image_name), image)
    elif psnr_q1 < psnr < psnr_q2:
        psnr_class_count[2] += 1
        cv2.imwrite(os.path.join(psnr_q1_2_path, image_name), image)
    elif psnr_lower < psnr < psnr_q1:
        psnr_class_count[1] += 1
        cv2.imwrite(os.path.join(psnr_lower_path, image_name), image)
    else:
        psnr_class_count[0] += 1
        cv2.imwrite(os.path.join(psnr_lower_xcpt_path, image_name), image)

    if ssim_upper < ssim:
        ssim_class_count[5] += 1
        cv2.imwrite(os.path.join(ssim_upper_xcpt_path, image_name), image)
    elif ssim_q3 < ssim < ssim_upper:
        ssim_class_count[4] += 1
        cv2.imwrite(os.path.join(ssim_upper_path, image_name), image)
    elif ssim_q2 < ssim < ssim_q3:
        ssim_class_count[3] += 1
        cv2.imwrite(os.path.join(ssim_q2_3_path, image_name), image)
    elif ssim_q1 < ssim < ssim_q2:
        ssim_class_count[2] += 1
        cv2.imwrite(os.path.join(ssim_q1_2_path, image_name), image)
    elif ssim_lower < ssim < ssim_q1:
        ssim_class_count[1] += 1
        cv2.imwrite(os.path.join(ssim_lower_path, image_name), image)
    else:
        ssim_class_count[0] += 1
        cv2.imwrite(os.path.join(ssim_lower_xcpt_path, image_name), image)
print()

psnr_upper_xcpt_pct = (psnr_class_count[5] / num_img) * 100
psnr_upper_pct = (psnr_class_count[4] / num_img) * 100
psnr_q2_3_pct= (psnr_class_count[3] / num_img) * 100
psnr_q1_2_pct = (psnr_class_count[2] / num_img) * 100
psnr_lower_pct = (psnr_class_count[1] / num_img) * 100
psnr_lower_xcpt_pct = (psnr_class_count[0] / num_img) * 100

ssim_upper_xcpt_pct = (ssim_class_count[5] / num_img) * 100
ssim_upper_pct = (ssim_class_count[4] / num_img) * 100
ssim_q2_3_pct= (ssim_class_count[3] / num_img) * 100
ssim_q1_2_pct = (ssim_class_count[2] / num_img) * 100
ssim_lower_pct = (ssim_class_count[1] / num_img) * 100
ssim_lower_xcpt_pct = (ssim_class_count[0] / num_img) * 100


classes = ['Outlier', 'Lower-Q1', 'Q1-Q2', 'Q2-Q3', 'Q3-Upper', 'Outlier']

print('Saving ' + dataset_name + '_PSNR_distribution_bar.png\t\t\t\t\t', end='\r')
plt.bar(classes, psnr_class_count, color='gold')
plt.title('Distribution based on PSNR')
plt.xlabel('Quartile range')
plt.ylabel('number ')
plt.savefig(os.path.join(result_path, dataset_name + '_PSNR_distribution_bar.png'))
plt.clf()

print('Saving ' + dataset_name + '_distribution.png\t\t\t\t\t', end='\r')
# set width of bar
barWidth = 0.25
# Set position of bar on X axis
bar1 = np.arange(len(psnr_class_count))
bar2 = [x + barWidth for x in bar1]
# Make the plot
plt.bar(bar1, psnr_class_count, color ='tomato', width = barWidth, label ='PSNR')
plt.bar(bar2, ssim_class_count, color ='gold', width = barWidth, label ='SSIM')
# Adding Xticks
plt.xlabel('Quartile range')
plt.ylabel('Number of images')
plt.title('Distribution based on PSNR and SSIM')
plt.xticks([r + barWidth for r in range(len(psnr_class_count))], labels=classes)
plt.legend()
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_distribution.png'))
plt.clf()

print('Saving ' + dataset_name + '_PSNR_distribution_pie.png\t\t\t\t\t', end='\r')
plt.pie(psnr_class_count, labels=classes, autopct='%.2f%%')
plt.title('Distribution based on PSNR')
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_PSNR_distribution_pie.png'))
plt.clf()

print('Saving ' + dataset_name + '_SSIM_distribution_pie.png\t\t\t\t\t', end='\r')
plt.pie(ssim_class_count, labels=classes, autopct='%.2f%%')
plt.title('Distribution based on SSIM')
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_SSIM_distribution_pie.png'))
plt.clf()
print()
'''

psnr_end_1 = math.ceil(sorted_psnr_results_1[-1])
psnr_start_1 = math.floor(sorted_psnr_results_1[0])
psnr_range_1 = range(psnr_start_1, psnr_end_1)
psnr_classes_count_1 = [i*0 for i in psnr_range_1]
psnr_classes_1 = []
psnr_classes_range_1 = []
for n in psnr_range_1:
    psnr_classes_1.append(str(n))
    psnr_classes_range_1.append(str(n)+'-'+str(n+1))
for psnr in psnr_results_1:
    psnr_classes_count_1[math.floor(psnr) - psnr_start_1] += 1
print('Saving ' + output_1_name + '_PSNR_bar.png\t\t\t\t\t', end='\r')
plt.bar(psnr_classes_1, psnr_classes_count_1, width=1.0, color='gold', edgecolor='black', align='edge')
plt.title(output_1_name + ' results distribution based on PSNR')
plt.xlabel('PSNR')
plt.ylabel('number of image')
plt.savefig(os.path.join(output_1_result_path, output_1_name + '_PSNR_bar.png'))
plt.clf()

psnr_end_2 = math.ceil(sorted_psnr_results_2[-1])
psnr_start_2 = math.floor(sorted_psnr_results_2[0])
psnr_range_2 = range(psnr_start_2, psnr_end_2)
psnr_classes_count_2 = [i*0 for i in psnr_range_2]
psnr_classes_2 = []
psnr_classes_range_2 = []
for n in psnr_range_2:
    psnr_classes_2.append(str(n))
    psnr_classes_range_2.append(str(n)+'-'+str(n+1))
for psnr in psnr_results_2:
    psnr_classes_count_2[math.floor(psnr) - psnr_start_2] += 1
print('Saving ' + output_2_name + '_PSNR_bar.png\t\t\t\t\t', end='\r')
plt.bar(psnr_classes_2, psnr_classes_count_2, width=1.0, color='gold', edgecolor='black', align='edge')
plt.title(output_2_name + ' results distribution based on PSNR')
plt.xlabel('PSNR')
plt.ylabel('number of image')
plt.savefig(os.path.join(output_2_result_path, output_2_name + '_PSNR_bar.png'))
plt.clf()

ssim_end_1 = math.ceil(sorted_ssim_results_1[-1] * 100)
ssim_start_1 = math.floor(sorted_ssim_results_1[0] * 100)
ssim_range_1 = range(ssim_start_1, ssim_end_1)
ssim_classes_count_1 = [i*0 for i in ssim_range_1]
ssim_classes_1 = []
ssim_classes_range_1 = []
for n in ssim_range_1:
    ssim_classes_1.append(f'{n/100:.2f}')
    ssim_classes_range_1.append(f'{n/100:.2f}-{(n+1)/100:.2f}')
for ssim in ssim_results_1:
    ssim_classes_count_1[math.floor(ssim * 100) - ssim_start_1] += 1
print('Saving ' + output_1_name + '_SSIM_bar.png\t\t\t\t\t', end='\r')
plt.bar(ssim_classes_1, ssim_classes_count_1,width=1.0, color='gold', edgecolor='black', align='edge')
plt.title(output_1_name + ' results distribution based on SSIM')
plt.xlabel('SSIM')
plt.ylabel('number of image')
plt.savefig(os.path.join(output_1_result_path, output_1_name + '_SSIM_bar.png'))
plt.clf()

ssim_end_2 = math.ceil(sorted_ssim_results_2[-1] * 100)
ssim_start_2 = math.floor(sorted_ssim_results_2[0] * 100)
ssim_range_2 = range(ssim_start_2, ssim_end_2)
ssim_classes_count_2 = [i*0 for i in ssim_range_2]
ssim_classes_2 = []
ssim_classes_range_2 = []
for n in ssim_range_2:
    ssim_classes_2.append(f'{n/100:.2f}')
    ssim_classes_range_2.append(f'{n/100:.2f}-{(n+1)/100:.2f}')
for ssim in ssim_results_2:
    ssim_classes_count_2[math.floor(ssim * 100) - ssim_start_2] += 1
print('Saving ' + output_2_name + '_SSIM_bar.png\t\t\t\t\t', end='\r')
plt.bar(ssim_classes_2, ssim_classes_count_2,width=1.0, color='gold', edgecolor='black', align='edge')
plt.title(output_2_name + ' results distribution based on SSIM')
plt.xlabel('SSIM')
plt.ylabel('number of image')
plt.savefig(os.path.join(output_2_result_path, output_2_name + '_SSIM_bar.png'))
plt.clf()

print()
