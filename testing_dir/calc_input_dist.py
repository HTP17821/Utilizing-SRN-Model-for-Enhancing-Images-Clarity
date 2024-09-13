import cv2
import skimage.metrics as metrics
import skimage.io as io
import os
import math
from matplotlib import pyplot as plt
import numpy as np

sharp_path = './GoPro_sharp'
blur_path = './GoPro_blur'
result_path = './input_analysis'
dataset_name = blur_path[2:-5]
dataset_result_path = os.path.join(result_path, dataset_name)

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(dataset_result_path):
    os.makedirs(dataset_result_path)

images = os.listdir(sharp_path)
num_img = len(images)

psnr_results = []
ssim_results = []
for i in range(num_img):
    image = images[i]
    print('Calculating metrics for image ' + str(i + 1) + '/' + str(num_img) + ' | ', end='')
    image_test = io.imread(os.path.join(blur_path, image))
    image_true = io.imread(os.path.join(sharp_path, image))
    psnr = metrics.peak_signal_noise_ratio(image_true, image_test)
    psnr_results.append(psnr)
    ssim = metrics.structural_similarity(im1=image_true, im2=image_test, channel_axis=2)
    ssim_results.append(ssim)
    print(f'PSNR = {psnr:.4f}, SSIM = {ssim:.4f}\t\t\t\t\t', end='\r')
print()

psnr_mean = sum(psnr_results) / len(psnr_results)
print(f'PSNR Average = {psnr_mean:.5f}')

ssim_mean = sum(ssim_results) / len(ssim_results)
print(f'SSIM Average = {ssim_mean:.5f}')

sorted_psnr_results = [psnr for psnr in psnr_results]
sorted_psnr_results.sort()
sorted_ssim_results = [ssim for ssim in ssim_results]
sorted_ssim_results.sort()

print('Calculating quartiles\t\t\t\t\t', end='\r')
if num_img % 2 == 0:
    psnr_q2 = (sorted_psnr_results[num_img // 2] + sorted_psnr_results[(num_img // 2) + 1]) / 2
    psnr_q1 = (sorted_psnr_results[(num_img // 2) // 2] + sorted_psnr_results[((num_img // 2) // 2) + 1]) / 2
    psnr_q3 = (sorted_psnr_results[(num_img // 2) + ((num_img // 2) // 2)] + sorted_psnr_results[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    psnr_iqr = psnr_q3 - psnr_q1
    psnr_lower = psnr_q1 - 1.5 * psnr_iqr
    psnr_upper = psnr_q3 + 1.5 * psnr_iqr
    psnr_max = sorted_psnr_results[-1]
    psnr_min = sorted_psnr_results[0]

    ssim_q2 = (sorted_ssim_results[num_img // 2] + sorted_ssim_results[(num_img // 2) + 1]) / 2
    ssim_q1 = (sorted_ssim_results[(num_img // 2) // 2] + sorted_ssim_results[((num_img // 2) // 2) + 1]) / 2
    ssim_q3 = (sorted_ssim_results[(num_img // 2) + ((num_img // 2) // 2)] + sorted_ssim_results[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    ssim_iqr = ssim_q3 - ssim_q1
    ssim_lower = ssim_q1 - 1.5 * ssim_iqr
    ssim_upper = ssim_q3 + 1.5 * ssim_iqr
    ssim_max = sorted_ssim_results[-1]
    ssim_min = sorted_ssim_results[0]
else:
    psnr_q2 = sorted_psnr_results[math.ceil(num_img / 2)]
    psnr_q1 = sorted_psnr_results[math.ceil(math.floor(num_img / 2) / 2)]
    psnr_q3 = sorted_psnr_results[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    psnr_iqr = psnr_q3 - psnr_q1
    psnr_lower = psnr_q1 - 1.5 * psnr_iqr
    psnr_upper = psnr_q3 + 1.5 * psnr_iqr
    psnr_max = sorted_psnr_results[-1]
    psnr_min = sorted_psnr_results[0]

    ssim_q2 = sorted_ssim_results[math.ceil(num_img / 2)]
    ssim_q1 = sorted_ssim_results[math.ceil(math.floor(num_img / 2) / 2)]
    ssim_q3 = sorted_ssim_results[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    ssim_iqr = ssim_q3 - ssim_q1
    ssim_lower = ssim_q1 - 1.5 * ssim_iqr
    ssim_upper = ssim_q3 + 1.5 * ssim_iqr
    ssim_max = sorted_ssim_results[-1]
    ssim_min = sorted_ssim_results[0]

print('Saving ' + dataset_name + '_quartiles.txt\t\t\t\t\t', end='\r')
file = open(os.path.join(dataset_result_path, dataset_name + '_quartiles.txt'), 'w')
file.write('============== PSNR ==============\n')
file.write('PSNR Mean: ' + f'{psnr_mean:.4f}\n')
file.write('PSNR Q2: ' + f'{psnr_q2:.4f}\n')
file.write('PSNR Q1: ' + f'{psnr_q1:.4f}\n')
file.write('PSNR Q3: ' + f'{psnr_q3:.4f}\n')
file.write('PSNR IQR: ' + f'{psnr_iqr:.4f}\n')
file.write('PSNR Lower: ' + f'{psnr_lower:.4f}\n')
file.write('PSNR Upper: ' + f'{psnr_upper:.4f}\n')
file.write('PSNR Min: ' + f'{psnr_min:.4f}\n')
file.write('PSNR Max: ' + f'{psnr_max:.4f}\n')
file.write('============== SSIM ==============\n')
file.write('SSIM Mean: ' + f'{ssim_mean:.4f}\n')
file.write('SSIM Q2: ' + f'{ssim_q2:.4f}\n')
file.write('SSIM Q1: ' + f'{ssim_q1:.4f}\n')
file.write('SSIM Q3: ' + f'{ssim_q3:.4f}\n')
file.write('SSIM IQR: ' + f'{ssim_iqr:.4f}\n')
file.write('SSIM Lower: ' + f'{ssim_lower:.4f}\n')
file.write('SSIM Upper: ' + f'{ssim_upper:.4f}\n')
file.write('SSIM Min: ' + f'{ssim_min:.4f}\n')
file.write('SSIM Max: ' + f'{ssim_max:.4f}\n')
file.close()

print('Saving ' + dataset_name + '_PSNR.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_psnr_results, meanline=True, showmeans=True)
plt.title('Distribution based on PSNR')
plt.ylabel('PSNR')
plt.grid(True)
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_PSNR.png'))
plt.clf()

print('Saving ' + dataset_name + '_SSIM.png\t\t\t\t\t', end='\r')
plt.boxplot(sorted_ssim_results, meanline=True, showmeans=True)
plt.title('Distribution based on SSIM')
plt.ylabel('SSIM')
plt.grid(True)
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_SSIM.png'))
plt.clf()

# PSNR_path = os.path.join(dataset_result_path, 'Sorted_by_PSNR')
# SSIM_path = os.path.join(dataset_result_path, 'Sorted_by_SSIM')
#
# psnr_upper_xcpt_path = os.path.join(PSNR_path, 'upper_xcpt')
# psnr_upper_path = os.path.join(PSNR_path, 'upper')
# psnr_q2_3_path = os.path.join(PSNR_path, 'q2_3')
# psnr_q1_2_path = os.path.join(PSNR_path, 'q1_2')
# psnr_lower_path = os.path.join(PSNR_path, 'lower')
# psnr_lower_xcpt_path = os.path.join(PSNR_path, 'lower_xcpt')
# psnr_class_count = [0, 0, 0, 0, 0, 0]
#
# ssim_upper_xcpt_path = os.path.join(SSIM_path, 'upper_xcpt')
# ssim_upper_path = os.path.join(SSIM_path, 'upper')
# ssim_q2_3_path = os.path.join(SSIM_path, 'q2_3')
# ssim_q1_2_path = os.path.join(SSIM_path, 'q1_2')
# ssim_lower_path = os.path.join(SSIM_path, 'lower')
# ssim_lower_xcpt_path = os.path.join(SSIM_path, 'lower_xcpt')
# ssim_class_count = [0, 0, 0, 0, 0, 0]
#
# if not os.path.exists(psnr_upper_xcpt_path):
#     os.makedirs(psnr_upper_xcpt_path)
# if not os.path.exists(psnr_upper_path):
#     os.makedirs(psnr_upper_path)
# if not os.path.exists(psnr_q2_3_path):
#     os.makedirs(psnr_q2_3_path)
# if not os.path.exists(psnr_q1_2_path):
#     os.makedirs(psnr_q1_2_path)
# if not os.path.exists(psnr_lower_path):
#     os.makedirs(psnr_lower_path)
# if not os.path.exists(psnr_lower_xcpt_path):
#     os.makedirs(psnr_lower_xcpt_path)
#
# if not os.path.exists(ssim_upper_xcpt_path):
#     os.makedirs(ssim_upper_xcpt_path)
# if not os.path.exists(ssim_upper_path):
#     os.makedirs(ssim_upper_path)
# if not os.path.exists(ssim_q2_3_path):
#     os.makedirs(ssim_q2_3_path)
# if not os.path.exists(ssim_q1_2_path):
#     os.makedirs(ssim_q1_2_path)
# if not os.path.exists(ssim_lower_path):
#     os.makedirs(ssim_lower_path)
# if not os.path.exists(ssim_lower_xcpt_path):
#     os.makedirs(ssim_lower_xcpt_path)
#
# for i in range(num_img):
#     print('Sorting images ' + str(i + 1) + '/' + str(num_img) + '\t\t\t\t\t', end='\r')
#     image_name = images[i]
#     image = cv2.imread(os.path.join(blur_path, image_name), flags=cv2.IMREAD_UNCHANGED)
#     psnr = psnr_results[i]
#     ssim = ssim_results[i]
#
#     if psnr_upper < psnr:
#         psnr_class_count[5] += 1
#         cv2.imwrite(os.path.join(psnr_upper_xcpt_path, image_name), image)
#     elif psnr_q3 < psnr < psnr_upper:
#         psnr_class_count[4] += 1
#         cv2.imwrite(os.path.join(psnr_upper_path, image_name), image)
#     elif psnr_q2 < psnr < psnr_q3:
#         psnr_class_count[3] += 1
#         cv2.imwrite(os.path.join(psnr_q2_3_path, image_name), image)
#     elif psnr_q1 < psnr < psnr_q2:
#         psnr_class_count[2] += 1
#         cv2.imwrite(os.path.join(psnr_q1_2_path, image_name), image)
#     elif psnr_lower < psnr < psnr_q1:
#         psnr_class_count[1] += 1
#         cv2.imwrite(os.path.join(psnr_lower_path, image_name), image)
#     else:
#         psnr_class_count[0] += 1
#         cv2.imwrite(os.path.join(psnr_lower_xcpt_path, image_name), image)
#
#     if ssim_upper < ssim:
#         ssim_class_count[5] += 1
#         cv2.imwrite(os.path.join(ssim_upper_xcpt_path, image_name), image)
#     elif ssim_q3 < ssim < ssim_upper:
#         ssim_class_count[4] += 1
#         cv2.imwrite(os.path.join(ssim_upper_path, image_name), image)
#     elif ssim_q2 < ssim < ssim_q3:
#         ssim_class_count[3] += 1
#         cv2.imwrite(os.path.join(ssim_q2_3_path, image_name), image)
#     elif ssim_q1 < ssim < ssim_q2:
#         ssim_class_count[2] += 1
#         cv2.imwrite(os.path.join(ssim_q1_2_path, image_name), image)
#     elif ssim_lower < ssim < ssim_q1:
#         ssim_class_count[1] += 1
#         cv2.imwrite(os.path.join(ssim_lower_path, image_name), image)
#     else:
#         ssim_class_count[0] += 1
#         cv2.imwrite(os.path.join(ssim_lower_xcpt_path, image_name), image)
# print()
#
# psnr_upper_xcpt_pct = (psnr_class_count[5] / num_img) * 100
# psnr_upper_pct = (psnr_class_count[4] / num_img) * 100
# psnr_q2_3_pct= (psnr_class_count[3] / num_img) * 100
# psnr_q1_2_pct = (psnr_class_count[2] / num_img) * 100
# psnr_lower_pct = (psnr_class_count[1] / num_img) * 100
# psnr_lower_xcpt_pct = (psnr_class_count[0] / num_img) * 100
#
# ssim_upper_xcpt_pct = (ssim_class_count[5] / num_img) * 100
# ssim_upper_pct = (ssim_class_count[4] / num_img) * 100
# ssim_q2_3_pct= (ssim_class_count[3] / num_img) * 100
# ssim_q1_2_pct = (ssim_class_count[2] / num_img) * 100
# ssim_lower_pct = (ssim_class_count[1] / num_img) * 100
# ssim_lower_xcpt_pct = (ssim_class_count[0] / num_img) * 100
#
#
# classes = ['Outlier', 'Lower-Q1', 'Q1-Q2', 'Q2-Q3', 'Q3-Upper', 'Outlier']
#
# print('Saving ' + dataset_name + '_PSNR_distribution_bar.png\t\t\t\t\t', end='\r')
# plt.bar(classes, psnr_class_count, color='gold')
# plt.title('Distribution based on PSNR')
# plt.xlabel('Quartile range')
# plt.ylabel('number ')
# plt.savefig(os.path.join(result_path, dataset_name + '_PSNR_distribution_bar.png'))
# plt.clf()
#
# print('Saving ' + dataset_name + '_distribution.png\t\t\t\t\t', end='\r')
# # set width of bar
# barWidth = 0.25
# # Set position of bar on X axis
# bar1 = np.arange(len(psnr_class_count))
# bar2 = [x + barWidth for x in bar1]
# # Make the plot
# plt.bar(bar1, psnr_class_count, color ='tomato', width = barWidth, label ='PSNR')
# plt.bar(bar2, ssim_class_count, color ='gold', width = barWidth, label ='SSIM')
# # Adding Xticks
# plt.xlabel('Quartile range')
# plt.ylabel('Number of images')
# plt.title('Distribution based on PSNR and SSIM')
# plt.xticks([r + barWidth for r in range(len(psnr_class_count))], labels=classes)
# plt.legend()
# plt.savefig(os.path.join(dataset_result_path, dataset_name + '_distribution.png'))
# plt.clf()
#
# print('Saving ' + dataset_name + '_PSNR_distribution_pie.png\t\t\t\t\t', end='\r')
# plt.pie(psnr_class_count, labels=classes, autopct='%.2f%%')
# plt.title('Distribution based on PSNR')
# plt.savefig(os.path.join(dataset_result_path, dataset_name + '_PSNR_distribution_pie.png'))
# plt.clf()
#
# print('Saving ' + dataset_name + '_SSIM_distribution_pie.png\t\t\t\t\t', end='\r')
# plt.pie(ssim_class_count, labels=classes, autopct='%.2f%%')
# plt.title('Distribution based on SSIM')
# plt.savefig(os.path.join(dataset_result_path, dataset_name + '_SSIM_distribution_pie.png'))
# plt.clf()
# print()

psnr_end = math.ceil(sorted_psnr_results[-1])
psnr_start = math.floor(sorted_psnr_results[0])
psnr_range = range(psnr_start, psnr_end)
psnr_classes_count = [i*0 for i in psnr_range]
psnr_classes = []
psnr_classes_range = []
for n in psnr_range:
    psnr_classes.append(str(n))
    psnr_classes_range.append(str(n)+'-'+str(n+1))
for psnr in psnr_results:
    psnr_classes_count[math.floor(psnr) - psnr_start] += 1
print('Saving ' + dataset_name + '_PSNR_distribution.png\t\t\t\t\t', end='\r')
plt.bar(psnr_classes, psnr_classes_count,width=1.0, color='gold', edgecolor='black', align='edge')
plt.title('Distribution based on PSNR')
plt.xlabel('PSNR')
plt.ylabel('number of image')
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_PSNR_distribution.png'))
plt.clf()

ssim_end = math.ceil(sorted_ssim_results[-1] * 100)
ssim_start = math.floor(sorted_ssim_results[0] * 100)
ssim_range = range(ssim_start, ssim_end)
ssim_classes_count = [i*0 for i in ssim_range]
ssim_classes = []
ssim_classes_range = []
for n in ssim_range:
    ssim_classes.append(f'{n/100:.2f}')
    ssim_classes_range.append(f'{n/100:.2f}-{(n+1)/100:.2f}')
for ssim in ssim_results:
    ssim_classes_count[math.floor(ssim * 100) - ssim_start] += 1
print('Saving ' + dataset_name + '_SSIM_distribution.png\t\t\t\t\t', end='\r')
plt.bar(ssim_classes, ssim_classes_count,width=1.0, color='gold', edgecolor='black', align='edge')
plt.title('Distribution based on SSIM')
plt.xlabel('SSIM')
plt.ylabel('number of image')
plt.savefig(os.path.join(dataset_result_path, dataset_name + '_SSIM_distribution.png'))
plt.clf()

PSNR_path = os.path.join(dataset_result_path, 'Sorted_by_PSNR')
SSIM_path = os.path.join(dataset_result_path, 'Sorted_by_SSIM')
for psnr_class_range in psnr_classes_range:
    class_path = os.path.join(PSNR_path, psnr_class_range)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
for ssim_class_range in ssim_classes_range:
    class_path = os.path.join(SSIM_path, ssim_class_range)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

for i in range(num_img):
    print('Sorting images ' + str(i + 1) + '/' + str(num_img) + '\t\t\t\t\t', end='\r')
    image_name = images[i]
    image = cv2.imread(os.path.join(blur_path, image_name), flags=cv2.IMREAD_UNCHANGED)
    psnr = psnr_results[i]
    ssim = ssim_results[i]

    psnr_class_path = os.path.join(PSNR_path, psnr_classes_range[math.floor(psnr) - psnr_start])
    cv2.imwrite(os.path.join(psnr_class_path, image_name), image)
    ssim_class_path = os.path.join(SSIM_path, ssim_classes_range[math.floor(ssim * 100) - ssim_start])
    cv2.imwrite(os.path.join(ssim_class_path, image_name), image)
print()
