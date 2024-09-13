import streamlit as st
import base64

st.set_page_config(
    page_title="GitHub Project",
    page_icon="globe_with_meridians",
    initial_sidebar_state='collapsed',
    layout='centered'
)

st.markdown(
    '''
    <style>
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    #MainMenu {
        visibility: hidden;
        height: 0%;
    }
    header {
        visibility: hidden;
        height: 0%;
    }
    footer {
        visibility: hidden;
        height: 0%;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

@st.cache_data
def get_img_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# img = get_img_base64("Images/gh.png")
# img2 = get_img_base64("Images/sb.png")
#
#
# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] {{
# background-image: url("data:image/png;base64,{img}");
# background-size: cover;
# background-position: center;
# }}
# """
# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img2}");
# background-size: cover;
# background-position: center;
# }}
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)


def home():
    st.markdown("""
        <style>
            .header {
                text-align: center;
                font-size: 100px;
                font-weight: bold;
                color: black; /* Màu xanh lá */
                background-color: white;
                margin-top: 0px;
                margin-bottom: 30px;
                border: none;
                border-radius: 0px;
            }
            .subheader {
                text-align: center;
                font-size: 60px;
                font-weight: bold;
                color: #ffffff; /* Màu trắng */
                margin-top: 20px;  
                margin-bottom: 20px; 
            }
            .description {
                text-align: center;
                font-size: 30px;
                color: #ffffff; /* Màu trắng */
                margin-top: 0px;
                margin-bottom: 20px; 
            }
            .button-container {
                display: flex;
                justify-content: center;
                margin-top: 0px; 
                margin-bottom: 0px; 
                color: black;
            }
            .button {
                white-space: nowrap;
                background-color: white;      /* Màu đen cho nền nút */
                color: black !important;      
                padding: 20px 50px;             /* dài rộng các button */
                font-size: 25px;
                border: none;
                border-radius: 10px;            /* bo viền button */
                cursor: pointer;
                margin: 40px;                   /* khoảng cách ngang giữa các button */
                text-decoration: none;
                text-align: center;
            }
            .button:hover {
                background-color: black; /* Màu hiệu ứng button */
                color: white !important;
                text-decoration: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">GitHub</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Click the button below to visit our GitHub repository to check out the code.</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="button-container">
            <a href="https://github.com/punpun0508/SRN_Deblur_fork" class="button" target="_blank">Visit GitHub Repository</a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subheader">Github of Members</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="button-container">
            <a href="https://github.com/HTP17821" class="button" target="_blank">Huỳnh Tấn Phát</a>
            <a href="https://github.com/punpun0508" class="button" target="_blank">Nguyễn Hoàng Việt</a>
            <a href="https://github.com/mochamadness" class="button" target="_blank">Nguyễn Vũ Dũng</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    home()
