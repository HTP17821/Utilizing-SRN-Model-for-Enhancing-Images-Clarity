import streamlit as st
from datetime import datetime
from Tab import Introduce, Convert, Compare, About
import base64
import time

# Cấu hình trang phải được đặt đầu tiên
st.set_page_config(
    page_title='SRN Deblur',
    page_icon='globe_with_meridians',
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


# img = get_img_base64("./Images/bg.png")
# img2 = get_img_base64("./Images/sb.png")

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] {{
# background-image: url("data:image/png;base64,{img}");
# background-size: cover;
# background-position: center;
# }}
#
# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img2}");
# background-size: cover;
# background-position: center;
# }}
#
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

# Title mới vào
st.title(':rainbow[SRN Deblur]')


# Hàm cập nhật thời gian thực
def display_time():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")  # Chỉ hiển thị giờ và phút
    st.subheader(f":date: :gray[Today is:  **{current_date}**  and Now is:  **{current_time}**] :clock3:")


# Tạo chỗ trống cho thời gian thực
time_placeholder = st.empty()
# Title mới vào
st.title('Sharpen blurred images')

# Define the tabs
tab1, tab2, tab3, tab4 = st.tabs(["Introduce", "Convert", "Compare", "About"])

# Tab 1 content
with tab1:
    Introduce.home()

# Tab 2 content
with tab2:
    Convert.home()

# Tab 3 content
with tab3:
    Compare.home()

# Tab 4 content
with tab4:
    About.home()

# Real-time clock update
while True:
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    time_placeholder.subheader(f":date: :gray[ **{current_date}** / **{current_time}**] :clock3:", divider='rainbow')
    
    time.sleep(1)
