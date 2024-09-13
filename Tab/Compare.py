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
    st.markdown('<div class="centered-content"><h3>Convert blurry image to sharp image</h3></div>',
                unsafe_allow_html=True)

    # Container cho upload ảnh và kết quả
    # st.markdown('<div class="container">', unsafe_allow_html=True)

    # Container bên trái để upload ảnh
    # st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    # st.image([image, image])

    uploaded_file = st.file_uploader("Upload blurry image", key='compare_tab_file_upload', type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Hiển thị ảnh đã upload
        image_name = uploaded_file.name
        files = {'image': (image_name, uploaded_file.getvalue(), 'multipart/form-data')}
        api_response = requests.post(url='http://localhost:8000/upload', files=files)
        uploaded_file = BytesIO(api_response.content)
        st.image(uploaded_file, caption='Original Image')

    st.markdown('Press "Deblur" to begin:')
    convert_button = st.button('Deblur', key='compare-tab-rainbow-button')
    deblur_done = False
    # Deblur
    if convert_button and uploaded_file is not None:
        with st.spinner("Deblurring image..."):
            files = {'image': (image_name, uploaded_file.getvalue(), 'multipart/form-data')}
            api_response = requests.post(url='http://localhost:8000/SRN_Deblur', files=files)
            st.image(BytesIO(api_response.content), caption='Modified SRN Deblur result')
            api_response = requests.get(url=f'http://localhost:8000/SRN_Deblur_Base/{image_name}')
            st.image(BytesIO(api_response.content), caption='Base SRN Deblur result')
            api_response = requests.get(url=f'http://localhost:8000/DeblurGAN/{image_name}')
            st.image(BytesIO(api_response.content), caption='DeblurGAN result')
            api_response = requests.get(url=f'http://localhost:8000/DeepDeblur/{image_name}')
            st.image(BytesIO(api_response.content), caption='DeepDeblur result')
        deblur_done = True

    # Container bên phải để hiển thị kết quả
    # st.markdown('<div class="result-container">', unsafe_allow_html=True)
    # st.write("The deblurring result will be displayed here.")
    # st.markdown("Original image will be on the left, deblurred image will be on the right")
    # render image-comparison

    @st.experimental_fragment
    def show_download_button(data, key):
        st.download_button(
            label="Download Image as PNG",
            data=data,
            key=key,
            file_name="result.png",
            mime="image/png"
        )

    if deblur_done:
        # with st.spinner("Processing..."):
        #     time.sleep(3)
        # Hiển thị ảnh kết quả
        # st.write("Deblurring result:")
        # st.image(BytesIO(api_response.content), caption="Deblurred Image", use_column_width=True)
        # add a download button
        # st.markdown('</div>', unsafe_allow_html=True)
        # st.markdown('</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["SRN Deblur Modified", "SRN Deblur Base", "DeepDeblur", "DeblurGAN", "All"])
        with tab1:
            with st.spinner("Loading comparison"):
                image_name = image_name[0:image_name.rfind('.')] + '.png'
                image_comparison(
                    img1=f'http://localhost:8000/blur/{image_name}',
                    img2=f'http://localhost:8000/result/{image_name}',
                    label1="Blurry Image",
                    label2="Modified SRN Deblur Output",
                )
            # Process image for calculation
            blur_api_response = requests.get(url=f'http://localhost:8000/blur/{image_name}')
            modified_srn_deblur_result_api_response_1 = requests.get(url=f'http://localhost:8000/result/{image_name}')

            modified_srn_deblur_download_bytes = BytesIO(modified_srn_deblur_result_api_response_1.content).getvalue()
            show_download_button(modified_srn_deblur_download_bytes, 'tab1')

            blurry_image_bytes = np.asarray(bytearray(BytesIO(blur_api_response.content).read()), dtype=np.uint8)
            blurry_image = cv2.imdecode(blurry_image_bytes, 1)
            # Convert color from BGR to RGB
            blurry_image = cv2.cvtColor(blurry_image, cv2.COLOR_BGR2RGB)

            modified_srn_deblur_result_bytes_1 = np.asarray(bytearray(BytesIO(modified_srn_deblur_result_api_response_1.content).read()), dtype=np.uint8)
            modified_srn_deblur_result_1 = cv2.imdecode(modified_srn_deblur_result_bytes_1, 1)
            # Convert color from BGR to RGB
            modified_srn_deblur_result_1 = cv2.cvtColor(modified_srn_deblur_result_1, cv2.COLOR_BGR2RGB)

            with st.spinner("Calculating metrics"):
                st.markdown('# Numerical evaluation metrics')
                st.markdown('Numerical metrics between the original image and Modified SRN Deblur output:')
                metrics.numerical_metrics(blurry_image, modified_srn_deblur_result_1, name_1='Blurry Image', name_2='Modified SRN Deblur Output')
                st.markdown('# Visual evaluation metrics')
                st.markdown('Visual evaluation metrics between the original image and Modified SRN Deblur output:')
                metrics.visual_metrics(blurry_image, modified_srn_deblur_result_1, name_1='Blurry Image', name_2='Modified SRN Deblur Output')

        with tab2:
            with st.spinner("Loading comparison"):
                image_name = image_name[0:image_name.rfind('.')] + '.png'
                image_comparison(
                    img1=f'http://localhost:8000/blur/{image_name}',
                    img2=f'http://localhost:8000/result/SRN_Deblur_Base/{image_name}',
                    label1="Blurry Image",
                    label2="Base SRN Deblur Output",
                )
                image_comparison(
                    img1=f'http://localhost:8000/result/{image_name}',
                    img2=f'http://localhost:8000/result/SRN_Deblur_Base/{image_name}',
                    label1="Modified SRN Deblur Output",
                    label2="Base SRN Deblur Output",
                )

            modified_srn_deblur_result_api_response_2 = requests.get(url=f'http://localhost:8000/result/{image_name}')
            base_srn_deblur_result_api_response = requests.get(url=f'http://localhost:8000/result/SRN_Deblur_Base/{image_name}')

            base_srn_deblur_download_bytes = BytesIO(base_srn_deblur_result_api_response.content).getvalue()
            show_download_button(base_srn_deblur_download_bytes, 'tab2')

            modified_srn_deblur_result_bytes_2 = np.asarray(bytearray(BytesIO(modified_srn_deblur_result_api_response_2.content).read()), dtype=np.uint8)
            modified_srn_deblur_result_2 = cv2.imdecode(modified_srn_deblur_result_bytes_2, 1)
            # Convert color from BGR to RGB
            modified_srn_deblur_result_2 = cv2.cvtColor(modified_srn_deblur_result_2, cv2.COLOR_BGR2RGB)

            base_srn_deblur_result_bytes = np.asarray(bytearray(BytesIO(base_srn_deblur_result_api_response.content).read()), dtype=np.uint8)
            base_srn_deblur_result = cv2.imdecode(base_srn_deblur_result_bytes, 1)
            # Convert color from BGR to RGB
            base_srn_deblur_result = cv2.cvtColor(base_srn_deblur_result, cv2.COLOR_BGR2RGB)

            with st.spinner("Calculating metrics"):
                st.markdown('# Numerical evaluation metrics')
                st.markdown('Numerical metrics between Modified SRN Deblur and Base SRN Deblur output:')
                metrics.numerical_metrics(modified_srn_deblur_result_2, base_srn_deblur_result, name_1='Modified SRN Deblur Output', name_2='Base SRN Deblur Output')
                st.markdown('# Visual evaluation metrics')
                st.markdown('Visual evaluation metrics between Modified SRN Deblur and Base SRN Deblur output:')
                metrics.visual_metrics(modified_srn_deblur_result_2, base_srn_deblur_result, name_1='Modified SRN Deblur Output', name_2='Base SRN Deblur Output')

        with tab3:
            with st.spinner("Loading comparison"):
                image_name = image_name[0:image_name.rfind('.')] + '.png'
                image_comparison(
                    img1=f'http://localhost:8000/blur/{image_name}',
                    img2=f'http://localhost:8000/result/DeepDeblur/{image_name}',
                    label1="Blurry Image",
                    label2="DeepDeblur Output",
                )
                image_comparison(
                    img1=f'http://localhost:8000/result/{image_name}',
                    img2=f'http://localhost:8000/result/DeepDeblur/{image_name}',
                    label1="SRN Deblur Output",
                    label2="DeepDeblur Output",
                )

            modified_srn_deblur_result_api_response_3 = requests.get(url=f'http://localhost:8000/result/{image_name}')
            deepdeblur_result_api_response = requests.get(url=f'http://localhost:8000/result/DeepDeblur/{image_name}')

            deepdeblur_download_bytes = BytesIO(deepdeblur_result_api_response.content).getvalue()
            show_download_button(deepdeblur_download_bytes, 'tab3')

            modified_srn_deblur_result_bytes_3 = np.asarray(bytearray(BytesIO(modified_srn_deblur_result_api_response_3.content).read()), dtype=np.uint8)
            modified_srn_deblur_result_3 = cv2.imdecode(modified_srn_deblur_result_bytes_3, 1)
            # Convert color from BGR to RGB
            modified_srn_deblur_result_3 = cv2.cvtColor(modified_srn_deblur_result_3, cv2.COLOR_BGR2RGB)

            deepdeblur_result_bytes = np.asarray(bytearray(BytesIO(deepdeblur_result_api_response.content).read()), dtype=np.uint8)
            deepdeblur_result = cv2.imdecode(deepdeblur_result_bytes, 1)
            # Convert color from BGR to RGB
            deepdeblur_result = cv2.cvtColor(deepdeblur_result, cv2.COLOR_BGR2RGB)

            with st.spinner("Calculating metrics"):
                st.markdown('# Numerical evaluation metrics')
                st.markdown('Numerical metrics between Modified SRN Deblur and DeepDeblur output:')
                metrics.numerical_metrics(modified_srn_deblur_result_3, deepdeblur_result, name_1='Modified SRN Deblur Output', name_2='DeepDeblur Output')
                st.markdown('# Visual evaluation metrics')
                st.markdown('Visual evaluation metrics between Modified SRN Deblur and DeepDeblur output:')
                metrics.visual_metrics(modified_srn_deblur_result_3, deepdeblur_result, name_1='Modified SRN Deblur Output', name_2='DeepDeblur Output')

        with tab4:
            with st.spinner("Loading comparison"):
                image_name = image_name[0:image_name.rfind('.')] + '.png'
                image_comparison(
                    img1=f'http://localhost:8000/blur/{image_name}',
                    img2=f'http://localhost:8000/result/DeblurGAN/{image_name}',
                    label1="Blurry Image",
                    label2="DeblurGAN Output",
                )
                image_comparison(
                    img1=f'http://localhost:8000/result/{image_name}',
                    img2=f'http://localhost:8000/result/DeblurGAN/{image_name}',
                    label1="SRN Deblur Output",
                    label2="DeblurGAN Output",
                )

            modified_srn_deblur_result_api_response_4 = requests.get(url=f'http://localhost:8000/result/{image_name}')
            deblurgan_result_api_response = requests.get(url=f'http://localhost:8000/result/DeblurGAN/{image_name}')

            deblurgan_download_bytes = BytesIO(deblurgan_result_api_response.content).getvalue()
            show_download_button(deblurgan_download_bytes, 'tab4')

            modified_srn_deblur_result_bytes_4 = np.asarray(bytearray(BytesIO(modified_srn_deblur_result_api_response_4.content).read()), dtype=np.uint8)
            modified_srn_deblur_result_4 = cv2.imdecode(modified_srn_deblur_result_bytes_4, 1)
            # Convert color from BGR to RGB
            modified_srn_deblur_result_4 = cv2.cvtColor(modified_srn_deblur_result_4, cv2.COLOR_BGR2RGB)

            deblurgan_result_bytes = np.asarray(bytearray(BytesIO(deblurgan_result_api_response.content).read()), dtype=np.uint8)
            deblurgan_result = cv2.imdecode(deblurgan_result_bytes, 1)
            # Convert color from BGR to RGB
            deblurgan_result = cv2.cvtColor(deblurgan_result, cv2.COLOR_BGR2RGB)

            with st.spinner("Calculating metrics"):
                st.markdown('# Numerical evaluation metrics')
                st.markdown('Numerical metrics between Modified SRN Deblur and DeblurGAN output:')
                metrics.numerical_metrics(modified_srn_deblur_result_4, deblurgan_result, name_1='Modified SRN Deblur Output', name_2='DeblurGAN Output')
                st.markdown('# Visual evaluation metrics')
                st.markdown('Visual evaluation metrics between Modified SRN Deblur and DeblurGAN output:')
                metrics.visual_metrics(modified_srn_deblur_result_4, deblurgan_result, name_1='Modified SRN Deblur Output', name_2='DeblurGAN Output')

        with tab5:
            blur_api_response = requests.get(url=f'http://localhost:8000/blur/{image_name}')
            m_srn_api_response = requests.get(url=f'http://localhost:8000/result/{image_name}')
            b_srn_api_response = requests.get(url=f'http://localhost:8000/result/SRN_Deblur_Base/{image_name}')
            deepdeblur_api_response = requests.get(url=f'http://localhost:8000/result/DeepDeblur/{image_name}')
            deblurgan_api_response = requests.get(url=f'http://localhost:8000/result/DeblurGAN/{image_name}')

            blur_bytes = np.asarray(bytearray(BytesIO(blur_api_response.content).read()), dtype=np.uint8)
            blur = cv2.imdecode(blur_bytes, 1)
            # Convert color from BGR to RGB
            blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

            m_srn_bytes = np.asarray(bytearray(BytesIO(m_srn_api_response.content).read()), dtype=np.uint8)
            m_srn = cv2.imdecode(m_srn_bytes, 1)
            # Convert color from BGR to RGB
            m_srn = cv2.cvtColor(m_srn, cv2.COLOR_BGR2RGB)

            b_srn_bytes = np.asarray(bytearray(BytesIO(b_srn_api_response.content).read()), dtype=np.uint8)
            b_srn = cv2.imdecode(b_srn_bytes, 1)
            # Convert color from BGR to RGB
            b_srn = cv2.cvtColor(b_srn, cv2.COLOR_BGR2RGB)

            deepdeblur_bytes = np.asarray(bytearray(BytesIO(deepdeblur_api_response.content).read()), dtype=np.uint8)
            deepdeblur = cv2.imdecode(deepdeblur_bytes, 1)
            # Convert color from BGR to RGB
            deepdeblur = cv2.cvtColor(deepdeblur, cv2.COLOR_BGR2RGB)

            deblurgan_bytes = np.asarray(bytearray(BytesIO(deblurgan_api_response.content).read()), dtype=np.uint8)
            deblurgan = cv2.imdecode(deblurgan_bytes, 1)
            # Convert color from BGR to RGB
            deblurgan = cv2.cvtColor(deblurgan, cv2.COLOR_BGR2RGB)

            with st.spinner("Calculating metrics"):
                metrics.numerical_metrics_all(blur, m_srn, b_srn, deepdeblur, deblurgan)


if __name__ == "__main__":
    home()
