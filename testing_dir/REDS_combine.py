import cv2
import os
import math

output_blur_path = './combined\\blur'
output_sharp_path = './combined\\sharp'
if not os.path.exists(output_blur_path):
    os.makedirs(output_blur_path)
if not os.path.exists(output_sharp_path):
    os.makedirs(output_sharp_path)

train_blur = './train_blur'
train_sharp = './train_sharp'

folders = os.listdir(train_blur)

num_folders = len(folders)
for n in range(num_folders):
    if ((n + 1) % 2) == 0:
        folder = folders[n]
        blur_images_path = os.path.join(train_blur, folder)
        blur_images = os.listdir(blur_images_path)
        num_blur = len(blur_images)
        sharp_images_path = os.path.join(train_sharp, folder)
        sharp_images = os.listdir(sharp_images_path)
        num_sharp = len(sharp_images)
        img_total = num_blur + num_sharp

        folder_pct = math.floor(((n + 1) / num_folders) * 100)

        for x in range(num_blur):
            image = blur_images[x]
            if image.split('.')[1] == 'png' and 26 <= int(image.split('.')[0][-3:]) <= 75:
                blur = cv2.imread(os.path.join(blur_images_path, image), flags=cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(output_blur_path, folder + '_' + image), blur)
            print('Copying ' + 'from folder ' + str(n+1) + '/' + str(num_folders) + ' ' + str(x + 1) + '/' + str(img_total) + ' | ' + str(folder_pct) + '% completed\t\t\t\t', end='\r')

        for y in range(num_sharp):
            image = sharp_images[y]
            if image.split('.')[1] == 'png' and 26 <= int(image.split('.')[0][-3:]) <= 75:
                sharp = cv2.imread(os.path.join(sharp_images_path, image), flags=cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(output_sharp_path, folder + '_' + image), sharp)
            print('Copying ' + 'from folder ' + str(n+1) + '/' + str(num_folders) + ' ' + str(num_blur + y + 1) + '/' + str(img_total) + ' | ' + str(folder_pct) + '% completed\t\t\t\t', end='\r')

print()
print('Number of blur images = ' + str(len(os.listdir(output_blur_path))))
print('Number of sharp images = ' + str(len(os.listdir(output_sharp_path))))
