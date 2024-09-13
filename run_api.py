import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import cv2
import math

app = FastAPI()

blur_path = './input'
temp_path = './temp'
result_path = './output'
base_result_path = './output_base'
GAN_result_path = './DeblurGAN_output/experiment_name/test_latest/images'
DeepDeblur_result_path = './DeepDeblur_output'
assets_path = './assets'


@app.post("/upload")
async def upload_image(image: UploadFile):
    image_name = image.filename
    image_path_temp = os.path.join(temp_path, image_name)
    image_temp = open(image_path_temp, 'wb')
    image_temp.write(image.file.read())

    image_name = image_name[0:image_name.rfind('.')] + '.png'
    input_path = os.path.join(blur_path, image_name)
    image = cv2.imread(image_path_temp)

    h = image.shape[0]
    w = image.shape[1]
    num_pixel = h * w
    if num_pixel > (1280*720):
        new_h = int(math.floor(h * math.sqrt((1280*720) / num_pixel)))
        new_w = int(math.floor(w * math.sqrt((1280*720) / num_pixel)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(input_path, image)
    gan_input_path = os.path.join(blur_path, image_name[0:-4])
    if not os.path.exists(gan_input_path):
        os.makedirs(gan_input_path)
    cv2.imwrite(os.path.join(gan_input_path, image_name), image)

    return FileResponse(input_path)


@app.post("/SRN_Deblur")
async def deblur_srn(image: UploadFile):
    image_name = image.filename
    # image_path_temp = os.path.join(temp_path, image_name)
    # image_temp = open(image_path_temp, 'wb')
    # image_temp.write(image.file.read())
    #
    # image_name = image_name[0:image_name.rfind('.')] + '.png'
    # input_path = os.path.join(blur_path, image_name)
    # image = cv2.imread(image_path_temp)
    # cv2.imwrite(input_path, image)
    # gan_input_path = os.path.join(blur_path, image_name[0:-4])
    # if not os.path.exists(gan_input_path):
    #     os.makedirs(gan_input_path)
    # cv2.imwrite(os.path.join(gan_input_path, image_name), image)

    img_name = image_name[0:image_name.rfind('.')] + '.png'
    input_path = os.path.join(blur_path, img_name)
    image = cv2.imread(input_path)

    height = image.shape[0]
    width = image.shape[1]
    os.system(f'python run_model.py --phase=run --height={height} --width={width} --input_path="{input_path}" --output_path="{result_path}" --model=lstm --num_scales=4 --scale_factor=0.75 --num_steps=2577120')

    return FileResponse(os.path.join(result_path, img_name))


@app.get("/SRN_Deblur_Base/{image_name}")
async def deblur_srn_base(image_name: str):
    img_name = image_name[0:image_name.rfind('.')] + '.png'
    input_path = os.path.join(blur_path, img_name)
    image = cv2.imread(input_path)
    height = image.shape[0]
    width = image.shape[1]
    os.system(f'python run_model.py --phase=run --height={height} --width={width} --input_path="{input_path}" --output_path="{base_result_path}" --model=color --num_scales=3 --scale_factor=0.5 --num_steps=523000')
    return FileResponse(os.path.join(base_result_path, img_name))


@app.get("/DeblurGAN/{image_name}")
async def deblur_gan(image_name: str):
    img_name = image_name[0:image_name.rfind('.')]
    input_path = os.path.join(blur_path, img_name)
    os.system(f'python "D:/DeblurGAN/test.py" --dataroot "D:/SRN_Deblur_fork/{input_path[1:]}" --model test --dataset_mode single --learn_residual --resize_or_crop no_change')
    return FileResponse(os.path.join(GAN_result_path, img_name + '_fake_B.png'))


@app.get("/DeepDeblur/{image_name}")
async def deepdeblur(image_name: str):
    img_name = image_name[0:image_name.rfind('.')]
    input_path = os.path.join(blur_path, img_name)
    os.system(f'python "D:/DeepDeblur/src/main.py" --save_dir REDS_L1 --demo true --demo_input_dir "D:/SRN_Deblur_fork/{input_path[1:]}" --demo_output_dir "D:/SRN_Deblur_fork/DeepDeblur_output" --precision half')
    return FileResponse(os.path.join(DeepDeblur_result_path, img_name + '.png'))


@app.get("/blur/{image_name}")
async def get_blur_image(image_name: str):
    image_path = os.path.join(blur_path, image_name)
    return FileResponse(image_path)


@app.get("/result/{image_name}")
async def get_result_image(image_name: str):
    image_path = os.path.join(result_path, image_name)
    return FileResponse(image_path)


@app.get("/result/SRN_Deblur_Base/{image_name}")
async def get_result_base(image_name: str):
    image_path = os.path.join(base_result_path, image_name)
    return FileResponse(image_path)


@app.get("/result/DeblurGAN/{image_name}")
async def get_result_deblurgan(image_name: str):
    img_name = image_name[0:image_name.rfind('.')]
    image_path = os.path.join(GAN_result_path, img_name + '_fake_B.png')
    return FileResponse(image_path)


@app.get("/result/DeepDeblur/{image_name}")
async def get_result_deepdeblur(image_name: str):
    image_path = os.path.join(DeepDeblur_result_path, image_name)
    return FileResponse(image_path)


@app.get("/asset/{image_name}")
async def get_asset(image_name: str):
    image_path = os.path.join(assets_path, image_name)
    return FileResponse(image_path)
