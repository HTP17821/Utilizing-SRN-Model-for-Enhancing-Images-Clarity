import streamlit as st
import base64


@st.cache_data
def load_image_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image


def home():
    st.markdown("""
        <style>
            /* Định dạng tiêu đề chính với hiệu ứng rainbow */
            .header {
                text-align: center;
                font-size: 50px;
                font-weight: bold;
                background: White;
                -webkit-background-clip: text;
                color: transparent;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            /* Định dạng tiêu đề phụ với gạch chân và tô sáng */
            .subheader {
                text-align: center;
                font-size: 30px;
                font-weight: bold;
                border-radius: 20px;                /* bo viền button */
                text-decoration: none;         /* Gạch chân */
                background-color: yellow;           /* Tô sáng */
                color: #333333;                     /* Màu đen */
                margin-top: 10px;
                margin-bottom: 10px;
            }
            /* Định dạng nội dung với màu chữ trắng */
            .content {
                font-size: 18px;
                color: #ffffff;             /* Màu trắng */
                text-align: justify;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            /* Định dạng các bước hướng dẫn */
            .step {
                font-size: 20px;
                color: #ffffff;             /* Màu trắng */
                margin-top: 10px;
                margin-bottom: 10px;
            }
            /* Định dạng container hình ảnh */
            .image-container {
                display: flex;
                justify-content: center;
                margin-top: 50px;
                margin-bottom: 20px;
            }
            /* Định dạng kích thước hình ảnh */
            .image-container img {
                width: 500px;               /* Chỉnh kích thước ảnh tại đây */
                height: auto;               /* Giữ tỷ lệ hình ảnh */
            }
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề chính với màu rainbow
    st.markdown('<div class="header">Introduction</div>', unsafe_allow_html=True)
    # Nội dung giới thiệu
    st.markdown("""
        <div class="content">
            The SRN (Scale-recurrent Network) Model is a multi-scale deep learning approach to deblurring images .            <br><br>
            It is used to convert blurry images into sharper images.        </div>
    """, unsafe_allow_html=True)

    # Tiêu đề phụ với gạch chân và tô sáng
    st.markdown('<div class="subheader">User manual</div>', unsafe_allow_html=True)
    # Nội dung hướng dẫn sử dụng
    st.markdown("""
        <div class="content">
            To deblur an image, follow these steps:
        </div>
        <div class="step">
            1. Find the blurry image that you want to deblur.
        </div>
        <div class="step">
            2. Open the "Convert" tab.
        </div>
        <div class="step">
            3. Upload the blurry image.
        </div>
        <div class="step">
            4. Click the "Convert" button and wait for the processing.
        </div>
        <div class="step">
            5. Once completed, you can download the sharp image.
        </div>
    """, unsafe_allow_html=True)

    # Tiêu đề phụ với gạch chân và tô sáng
    st.markdown('<div class="subheader">Illustration image</div>', unsafe_allow_html=True)
    # Nội dung ví dụ minh họa
    st.markdown("""
        <div class="content">
            Below is an example of before and after results using the SRN model:
        </div>
    """, unsafe_allow_html=True)

    # Tải và hiển thị ảnh
    blurred_image_base64 = load_image_as_base64("./Images/blur.jpg")
    sharp_image_base64 = load_image_as_base64("./Images/sharp.jpg")
    st.markdown(f"""
        <div class="image-container">
            <img src="data:image/jpeg;base64,{blurred_image_base64}" alt="Blurred Image" style="margin-right: 20px;">
            <img src="data:image/jpeg;base64,{sharp_image_base64}" alt="Sharp Image">
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    home()
