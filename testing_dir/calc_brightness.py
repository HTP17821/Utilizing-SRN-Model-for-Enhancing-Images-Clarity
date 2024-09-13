import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageStat
import math

images_path = "./GoPro"
results_path = './brightness_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
result_images_path = os.path.join(results_path, images_path[2:])
if not os.path.exists(result_images_path):
    os.makedirs(result_images_path)

images = os.listdir(images_path)
num_img = len(images)

#gt_files = os.listdir("./gt")
#print(len(gt_files))

def brightness(img_file):
   img = Image.open(img_file)
   stat = ImageStat.Stat(img)
   r,g,b = stat.mean
   return math.sqrt(0.299*(r**2) + 0.587*(g**2) + 0.114*(b**2))

images_brightness = []
for i in range(num_img):
    print('Calculating perceived brightness ' + str(i + 1) + '/' + str(num_img) + '\t\t\t\t', end='\r')
    _image = images[i]
    image = os.path.join(images_path, _image)
    images_brightness.append(brightness(image))
print()
sorted_images_brightness = [b for b in images_brightness]
sorted_images_brightness.sort()

print('Calculating quartiles')
mean_brightness = sum(images_brightness) / len(images_brightness)
if num_img % 2 == 0:
    brightness_q2 = (sorted_images_brightness[num_img // 2] + sorted_images_brightness[(num_img // 2) + 1]) / 2
    brightness_q1 = (sorted_images_brightness[(num_img // 2) // 2] + sorted_images_brightness[((num_img // 2) // 2) + 1]) / 2
    brightness_q3 = (sorted_images_brightness[(num_img // 2) + ((num_img // 2) // 2)] + sorted_images_brightness[((num_img // 2) + ((num_img // 2) // 2)) + 1]) / 2
    q1_2 = [b for b in sorted_images_brightness[((num_img // 2) // 2) + 1 : (num_img // 2) + 1]]
    q1_2_mean = sum(q1_2) / len(q1_2)
else:
    brightness_q2 = sorted_images_brightness[math.ceil(num_img / 2)]
    brightness_q1 = sorted_images_brightness[math.ceil(math.floor(num_img / 2) / 2)]
    brightness_q3 = sorted_images_brightness[math.ceil(num_img / 2) + math.ceil(math.floor(num_img / 2) / 2)]
    q1_2 = [b for b in sorted_images_brightness[math.ceil(math.floor(num_img / 2) / 2) + 1 : math.ceil(num_img / 2)]]
    q1_2_mean = sum(q1_2) / len(q1_2)

print('Saving quartiles.txt')
file = open(os.path.join(results_path, 'quartiles.txt'), 'w')
file.write('Mean Brightness: ' + f'{mean_brightness:.4f}\n')
file.write('Brightness Q2: ' + f'{brightness_q2:.4f}\n')
file.write('Brightness Q1: ' + f'{brightness_q1:.4f}\n')
file.write('Brightness Q3: ' + f'{brightness_q3:.4f}\n')
file.write('Mean Brightness Q1-Q2: ' + f'{q1_2_mean:.4f}\n')
file.close()

images_class = []
for i in range(num_img):
    print('Sorting images ' + str(i + 1) + '/' + str(num_img) + '\t\t\t\t', end='\r')
    dark_images_path = os.path.join(result_images_path, 'dark')
    bright_images_path = os.path.join(result_images_path, 'bright')
    if not os.path.exists(dark_images_path):
        os.makedirs(dark_images_path)
    if not os.path.exists(bright_images_path):
        os.makedirs(bright_images_path)
    _image = images[i]
    if images_brightness[i] > q1_2_mean:
        image = cv2.imread(os.path.join(images_path, _image), flags=cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(bright_images_path, _image), image)
        images_class.append(1)
    else:
        image = cv2.imread(os.path.join(images_path, _image), flags=cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(dark_images_path, _image), image)
        images_class.append(0)
print()

classes = ['dark', 'bright']
class_count = [0, 0]
for image_class in images_class:
    class_count[image_class] += 1

print('Saving perceived_brightness.png')
plt.boxplot(images_brightness, meanline=True, showmeans=True)
plt.title('Image Perceived brightness')
plt.ylabel('brightness')
plt.grid(True)
plt.savefig(os.path.join(results_path, 'perceived_brightness.png'))
plt.clf()

print('Saving classes.png')
plt.bar(classes, class_count)
plt.title('Image Classes')
plt.ylabel('number of images')
plt.savefig(os.path.join(results_path, 'classes.png'))
plt.clf()
