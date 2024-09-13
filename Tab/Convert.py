# Convert.py
import streamlit as st
import base64
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison
from io import BytesIO
from Tab import metrics
import requests


# Hàm để tải ảnh lên và mã hóa base64
def load_image_as_base64(file):
    return base64.b64encode(file.read()).decode()


# Giao diện chính cho tab "Convert"
def home():
    css_markdown = """
        <style>
            .centered-content {
                display: flex;                    /* Thiết lập chế độ hiển thị thành flexbox */
                justify-content: center;          /* Căn giữa nội dung theo chiều ngang */
                align-items: center;              /* Căn giữa nội dung theo chiều dọc */
                flex-direction: column;           /* Sắp xếp các phần tử con theo cột */
            }
            .rainbow-button {
                font-size: 30px;                  /* Kích thước chữ của button */
                font-weight: bold;                /* Độ đậm chữ của button */
                background: white;
                border: none;                     /* Loại bỏ đường viền của button */
                color: white;                     /* Màu chữ của button */
                padding: 10px 100px;              /* rộng và dài nút button */
                text-align: center;               /* Căn giữa chữ bên trong button */
                text-decoration: none;            /* gạch chân */
                display: inline-block;            /* Hiển thị button dưới dạng khối nội tuyến */
                margin: 10px 10px;                /* Khoảng cách xung quanh button */
                cursor: pointer;                  /* Thay đổi con trỏ thành dạng pointer khi di chuột qua */
                border-radius: 10px;              /* Bo tròn các góc của button */
            }
            .rainbow-button:hover {
                background: gray;
            }
            .container {                          /* Thanh hiển thị ảnh đang load */
                display: flex;                    /* Thiết lập chế độ hiển thị thành flexbox */
                justify-content: space-between;   /* Căn đều khoảng cách giữa các phần tử con */
                margin: 0px 10px;                /* Khoảng cách xung quanh container */
            }
            .upload-container, .result-container {
                width: 50%;                       /* Chiều rộng của container */
                padding: 10px;                    /* Đệm bên trong container */
                border-radius: 20px;              /* Bo tròn các góc của container */
                background-color: rgba(255, 255, 255, 0.1);  /* Màu nền trong suốt nhẹ */
            }
            .image-preview {
                max-width: 100%;                  /* Chiều rộng tối đa của ảnh bằng 100% container */
                border-radius: 10px;              /* Bo tròn các góc của ảnh */
                margin-top: 10px;                 /* Khoảng cách phía trên của ảnh */
            }
        </style>
    """
    # CSS tùy chỉnh cho nút button 
    st.markdown("""
        <style>
            .centered-content {
                display: flex;                    /* Thiết lập chế độ hiển thị thành flexbox */
                justify-content: center;          /* Căn giữa nội dung theo chiều ngang */
                align-items: center;              /* Căn giữa nội dung theo chiều dọc */
                flex-direction: column;           /* Sắp xếp các phần tử con theo cột */
            }
            .stButton>button {
                font-size: 80px;                  /* Kích thước chữ của button */
                font-weight: bold;                /* Độ đậm chữ của button */
                background: white;
                border: none;                     /* Loại bỏ đường viền của button */
                color: black;                     /* Màu chữ của button */
                padding: 10px 100px;              /* rộng và dài nút button */
                text-align: center;               /* Căn giữa chữ bên trong button */
                text-decoration: none;            /* gạch chân */
                display: inline-block;            /* Hiển thị button dưới dạng khối nội tuyến */
                margin: 10px 10px;                /* Khoảng cách xung quanh button */
                cursor: pointer;                  /* Thay đổi con trỏ thành dạng pointer khi di chuột qua */
                border-radius: 10px;              /* Bo tròn các góc của button */
            }
            .stButton>button:hover {
                background: black;
                color: white;
            }
            .stButton {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                width: 100%;
            }

            .container {                          /* Thanh hiển thị ảnh đang load */
                display: flex;                    /* Thiết lập chế độ hiển thị thành flexbox */
                justify-content: space-between;   /* Căn đều khoảng cách giữa các phần tử con */
                margin: 0px 10px;                /* Khoảng cách xung quanh container */
            }
            .upload-container, .result-container {
                width: 50%;                       /* Chiều rộng của container */
                padding: 10px;                    /* Đệm bên trong container */
                border-radius: 20px;              /* Bo tròn các góc của container */
                background-color: rgba(255, 255, 255, 0.1);  /* Màu nền trong suốt nhẹ */
            }
            .image-preview {
                max-width: 100%;                  /* Chiều rộng tối đa của ảnh bằng 100% container */
                border-radius: 10px;              /* Bo tròn các góc của ảnh */
                margin-top: 10px;                 /* Khoảng cách phía trên của ảnh */
            }
        </style>
    """, unsafe_allow_html=True)

    # api_response = requests.get(url='http://localhost:8000/asset/picture.png')
    # image = BytesIO(api_response.content)
    image_name = 'picture.png'
    # api_response = requests.get(url='http://localhost:8000/asset/loading.gif')
    # deblurred_image = BytesIO(api_response.content)

    # Nội dung chào hỏi và nút "Convert"
    st.markdown('<div class="centered-content"><h3>Convert blurry image to sharp image</h3></div>', unsafe_allow_html=True)

    # Container cho upload ảnh và kết quả
    # st.markdown('<div class="container">', unsafe_allow_html=True)

    # Container bên trái để upload ảnh
    # st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    # st.image([image, image])

    uploaded_file = st.file_uploader("Upload blurry image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Hiển thị ảnh đã upload
        image_name = uploaded_file.name
        files = {'image': (image_name, uploaded_file.getvalue(), 'multipart/form-data')}
        api_response = requests.post(url='http://localhost:8000/upload', files=files)
        uploaded_file = BytesIO(api_response.content)
        st.image(uploaded_file, caption='Original Image')

    st.markdown('Press "Deblur" to begin:')
    convert_button = st.button('Deblur', key='rainbow-button')
    deblur_done = False
    # Deblur
    if convert_button and uploaded_file is not None:
        with st.spinner("Deblurring image..."):
            files = {'image': (image_name, uploaded_file.getvalue(), 'multipart/form-data')}
            api_response = requests.post(url='http://localhost:8000/SRN_Deblur', files=files)
            st.image(BytesIO(api_response.content), caption='SRN Deblur result')
        deblur_done = True

    # Container bên phải để hiển thị kết quả
    # st.markdown('<div class="result-container">', unsafe_allow_html=True)
    # st.write("The deblurring result will be displayed here.")
    # st.markdown("Original image will be on the left, deblurred image will be on the right")
    # render image-comparison

    @st.experimental_fragment
    def show_download_button(data):
        st.download_button(
            label="Download Image as PNG",
            data=data,
            file_name="result.png",
            mime="image/png"
        )

    if deblur_done:
        with st.spinner("Loading comparison"):
            image_name = image_name[0:image_name.rfind('.')] + '.png'
            image_comparison(
                img1=f'http://localhost:8000/blur/{image_name}',
                img2=f'http://localhost:8000/result/{image_name}',
                label1="Original",
                label2="Deblurred",
            )
        st.write("Press the button below to download the deblurred image.")
        image_bytes = BytesIO(api_response.content).getvalue()
        # add a download button
        show_download_button(image_bytes)
        # with st.spinner("Processing..."):
        #     time.sleep(3)
        # Hiển thị ảnh kết quả
        # st.write("Deblurring result:")
        # st.image(BytesIO(api_response.content), caption="Deblurred Image", use_column_width=True)
        # add a download button
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Process image for calculation
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # Convert color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        deblurred_bytes = np.asarray(bytearray(BytesIO(api_response.content).read()), dtype=np.uint8)
        deblurred_image = cv2.imdecode(deblurred_bytes, 1)
        # Convert color from BGR to RGB
        deblurred_image = cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB)
        with st.spinner("Calculating metrics"):
            st.markdown('# Numerical evaluation metrics')
            st.markdown('Numerical metrics between the original image and the deblurred image:')
            metrics.numerical_metrics(image, deblurred_image, name_1='Blurry Image', name_2='SRN Deblur Output')
            st.markdown('# Visual evaluation metrics')
            st.markdown('Visual evaluation metrics between the original image and the deblurred image:')
            metrics.visual_metrics(image, deblurred_image, name_1='Blurry Image', name_2='SRN Deblur Output')


if __name__ == "__main__":
    home()
