import skimage.metrics as metrics
import skimage.io as io
import os

gt_path = "./gt_selective"
result_path = "./results_mod"

if not os.path.exists(result_path):
    os.makedirs(result_path)

images = os.listdir(gt_path)
num_img = len(images)

#gt_files = os.listdir("./gt")
#print(len(gt_files))

psnr_total = 0
ssim_total = 0
for i in range(num_img):
    image = images[i]
    print(f'Calculating metrics for image {i+1:d}/{num_img:d}', end='')
    image_test = io.imread(os.path.join(result_path, image))
    image_true = io.imread(os.path.join(gt_path, image))
    psnr = metrics.peak_signal_noise_ratio(image_true, image_test)
    psnr_total += psnr
    ssim = metrics.structural_similarity(im1=image_true, im2=image_test, channel_axis=2)
    ssim_total += ssim
    psnr_avg = psnr_total / (i + 1)
    ssim_avg = ssim_total / (i + 1)
    # print(f' | PSNR = {psnr_avg:.5f}, SSIM = {ssim_avg:.5f}\t\t\t\t\t', end='\r')
    print(f' | PSNR = {psnr:.4f}, SSIM = {ssim:.4f}')
print()

# psnr_avg = psnr_total / num_img
# print("PSNR Average = " + str(psnr_avg))

# ssim_avg = ssim_total / num_img
# print("SSIM Average = " + str(ssim_avg))
