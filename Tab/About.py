import streamlit as st
import base64


@st.cache_data
def get_img_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def home():
    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .content {
                text-align: center;
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .mentor-container, .member-container {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                margin-bottom: 20px;
            }
            .mentor, .member {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .mentor .info, .member .info {
                width: 45%;
                text-align: center;
            }
            .image {
                display: flex;
                justify-content: center;
                margin-top: 50px;
                margin-bottom: 20px;
                width: 100px !important;
                height: auto;
            }

        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề
    st.markdown('<div class="title">About us</div>', unsafe_allow_html=True)
    
    # Nội dung
    # st.markdown("""
    #     <div class="content">
    #         We are a group of 3 members who are passionate about the field (machine learning / deep learning) and 2 mentors with extensive experience in this field. During the process of working on the final project,
    #         we spent a lot of enthusiasm and energy to create an AI model that contains all the knowledge we have accumulated during 4 years of study at FPT University. Below is the information. detailed information about each person.
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Thông tin mentor
    img = get_img_base64("./Images/avt.png")
    img1 = get_img_base64("./Images/avt.png")
    
    st.markdown('<div class="mentor-container">', unsafe_allow_html=True)
    st.markdown('<div class="title">Mentors</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mentor">', unsafe_allow_html=True)
    if img:
        st.markdown(f'<img src="data:image/jpeg;base64,{img}" alt="Mentor 1 Avatar" class="image">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="data:image/jpeg;base64,{"path_to_placeholder.jpg"}" alt="Placeholder Avatar" class="image">', unsafe_allow_html=True)
    st.markdown('<div class="info">Mr. Nguyễn Quốc Tiến<br>mentor1@gmail.com</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mentor">', unsafe_allow_html=True)
    if img1:
        st.markdown(f'<img src="data:image/jpeg;base64,{img1}" alt="Mentor 2 Avatar" class="image">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="data:image/jpeg;base64,{"path_to_placeholder.jpg"}" alt="Placeholder Avatar" class="image">', unsafe_allow_html=True)
    st.markdown('<div class="info">Ms. Lâm Khả Hân<br>mentor2@gmail.com</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Thông tin thành viên
    img2 = get_img_base64("Images/avt.png")
    img3 = get_img_base64("Images/avt.png")
    img4 = get_img_base64("Images/avt.png")
    
    # --------------------------------------------------------------------------------------------------------
    st.markdown('<div class="member-container">', unsafe_allow_html=True)
    st.markdown('<div class="title">Members</div>', unsafe_allow_html=True)
    # Member 1
    st.markdown('<div class="member">', unsafe_allow_html=True)
    if img2:
        st.markdown(f'<img src="data:image/jpeg;base64,{img2}" alt="Member 1 Avatar" class="image">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="data:image/jpeg;base64,{"path_to_placeholder.jpg"}" alt="Placeholder Avatar" class="image">', unsafe_allow_html=True)
    st.markdown('<div class="info">Huỳnh Tấn Phát<br>SE150054<br>email</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Member 2
    st.markdown('<div class="member">', unsafe_allow_html=True)
    if img3:
        st.markdown(f'<img src="data:image/jpeg;base64,{img4}" alt="Member 3 Avatar" class="image">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="data:image/jpeg;base64,{"path_to_placeholder.jpg"}" alt="Placeholder Avatar" class="image">', unsafe_allow_html=True)
    st.markdown('<div class="info">Nguyễn Hoàng Việt<br>SE160968<br>email</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Member 3
    st.markdown('<div class="member">', unsafe_allow_html=True)
    if img4:
        st.markdown(f'<img src="data:image/jpeg;base64,{img4}" alt="Member 3 Avatar" class="image">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="data:image/jpeg;base64,{"path_to_placeholder.jpg"}" alt="Placeholder Avatar" class="image">', unsafe_allow_html=True)
    st.markdown('<div class="info">Nguyễn Vũ Dũng<br>SE150719<br>email</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    # --------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    home()
