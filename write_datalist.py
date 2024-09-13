import os

train_path = './training_set'

paths = os.listdir(train_path)
#for p in os.listdir(train_path):
    #paths.append(os.path.join(train_path, p))

blur_images = []
sharp_images = []
for path in paths:
    blur_path = path + '/' + 'blur'
    sharp_path = path + '/' + 'sharp'
    for file in os.listdir(train_path + '/' + blur_path):
        if file.split('.')[1] == 'png':
            blur_images.append(blur_path + '/' + file)
    for file in os.listdir(train_path + '/' + sharp_path):
        if file.split('.')[1] == 'png':
            sharp_images.append(sharp_path + '/' +file)

file = open('datalist.txt', 'w')
for i in range(len(sharp_images)):
    line = sharp_images[i] + ' ' + blur_images[i] + '\n'
    print(line)
    file.write(line)
