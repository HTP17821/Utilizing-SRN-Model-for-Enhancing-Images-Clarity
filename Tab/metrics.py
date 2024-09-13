import streamlit as st
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import pywt
import pandas as pd

import matplotlib.pyplot as plt
# from skimage.feature import hog
# from skimage import data, exposure

from streamlit_image_comparison import image_comparison


@st.experimental_fragment
def show_download_button(data):
    st.download_button(
        label="Download Image as PNG",
        data=data,
        file_name="result.png",
        mime="image/png"
    )


# numerical metrics
def calculate_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def calculate_tenengrad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.sqrt(sobelx**2 + sobely**2).mean()
    return tenengrad


def calculate_psi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contrast = gray.std()
    psi = edges.mean() * contrast
    return psi


# def calculate_gradient_magnitude(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
#     return gradient_magnitude


def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray)
    return entropy


def calculate_wavelet_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(gray, 'db1', level=3)
    # Take only the detail coefficients
    detail_coeffs = coeffs[1:]
    sharpness = sum([np.sum(np.abs(d)) for d in detail_coeffs])
    return sharpness


def calculate_all_image(image):
    laplacian_variance = calculate_laplacian_variance(image)
    tenengrad = calculate_tenengrad(image)
    psi = calculate_psi(image)
    # gradient_magnitude = calculate_gradient_magnitude(image)
    entropy = calculate_entropy(image)
    wavelet_sharpness = calculate_wavelet_sharpness(image)
    return laplacian_variance, tenengrad, psi, entropy, wavelet_sharpness


# visual metrics
def plot_gradient_histogram_overlay(image1, image2, label1='Blur image', label2='Deblurred image'):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(sobelx1**2 + sobely1**2)
    
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(sobelx2**2 + sobely2**2)
    plt.figure(figsize=(10, 5))
    plt.hist(gradient_magnitude1.ravel(), bins=256, range=(0.0, 255.0), fc='dodgerblue', ec='dodgerblue', alpha=1.0, label=label1)
    plt.hist(gradient_magnitude2.ravel(), bins=256, range=(0.0, 255.0), fc='tomato', ec='tomato', alpha=0.5, label=label2)
    
    plt.title('Gradient Magnitude Histogram')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(plt.gcf())


def plot_intensity_histogram_overlay(image1, image2, label1='Blur image', label2='Deblurred image'):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(20, 5))
    plt.hist(gray1.ravel(), bins=256, range=(0, 256), fc='dodgerblue', ec='dodgerblue', alpha=1.0, label=label1)
    plt.hist(gray2.ravel(), bins=256, range=(0, 256), fc='tomato', ec='tomato', alpha=0.5, label=label2)
    
    plt.title('Intensity Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(plt.gcf())


def plot_edge_map(image1, image2, label1='Blur image', label2='Deblurred image'):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray1, 100, 200)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    edges2 = cv2.Canny(gray2, 100, 200)

    image_comparison(
            img1=edges1,
            img2=edges2,
            label1=label1,
            label2=label2,
        )


# def plot_hog(image1, image2, label1='Blur image', label2='Deblurred image'):
#
#     fd1, hog_image1 = hog(
#         image1,
#         orientations=8,
#         pixels_per_cell=(16, 16),
#         cells_per_block=(1, 1),
#         visualize=True,
#         channel_axis=-1,
#     )
#     fd2, hog_image2 = hog(
#         image2,
#         orientations=8,
#         pixels_per_cell=(16, 16),
#         cells_per_block=(1, 1),
#         visualize=True,
#         channel_axis=-1,
#     )
#
#     ig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 5), sharex=True, sharey=True)
#     hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 20))
#
#     ax1.axis('off')
#     ax1.imshow(hog_image_rescaled1, cmap=plt.cm.gray)
#     ax1.set_title(label1)
#
#     # Rescale histogram for better display
#     hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 20))
#
#     ax2.axis('off')
#     ax2.imshow(hog_image_rescaled2, cmap=plt.cm.gray)
#     ax2.set_title(label2)
#     st.pyplot(plt.gcf())


def equalize_colored_image(image):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Equalize the Y channel (luminance)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    # Convert back to BGR
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return equalized_image


def plot_histogram_equalization(image):
    equalized = equalize_colored_image(image)
    image_comparison(
            img1=image,
            img2=equalized,
            label1="Deblurred Image",
            label2="Histogram equalized",
        )


def numerical_metrics(blur, deblurred, name_1, name_2):
    blur_metrics = calculate_all_image(blur)
    deblur_metrics = calculate_all_image(deblurred)

    # Create a dictionary to hold the data
    data = {
        'Image': [name_1, name_2, 'Metrics'],
        'Laplacian Variance': [blur_metrics[0], deblur_metrics[0], '- Measures the variance of the Laplacian (second derivative) of the image\n- Higher is better'],
        'Tenengrad Method': [blur_metrics[1], deblur_metrics[1], '- Calculates the gradient magnitude using Sobel operators\n- Higher is better'],
        'Perceptual Sharpness Index (PSI)': [blur_metrics[2], deblur_metrics[2], '- Combines edge detection and contrast measures\n- Higher is better'],
        'Entropy-Based Method': [blur_metrics[3], deblur_metrics[3], '- Measures the amount of information in the image using entropy\n- Higher is better'],
        'Wavelet Transform Based Method': [blur_metrics[4], deblur_metrics[4], '- Decomposes the image into different frequency bands\n- Sharpness is assessed based on the high-frequency components']
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Image')  # Assuming 'Image' column contains 'Blur' and 'Sharp' values
    df = df.transpose()  # transposes the dataframe, effectively flipping it vertically
    df['Metrics'] = df['Metrics'].apply(lambda x: x.replace('\n', '<br>') if isinstance(x, str) else x)
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)


def numerical_metrics_all(blur, m_srn, b_srn, deepdeblur, deblurgan):
    blur_metrics = calculate_all_image(blur)
    m_srn_metrics = calculate_all_image(m_srn)
    b_srn_metrics = calculate_all_image(b_srn)
    deepdeblur_metrics = calculate_all_image(deepdeblur)
    deblurgan_metrics = calculate_all_image(deblurgan)

    # Create a dictionary to hold the data
    data = {
        'Image': ['Input', 'Modified SRN Deblur', 'Base SRN Deblur', 'DeepDeblur', 'DeblurGAN', 'Metrics'],
        'Laplacian Variance': [blur_metrics[0], m_srn_metrics[0], b_srn_metrics[0], deepdeblur_metrics[0], deblurgan_metrics[0], '- Measures the variance of the Laplacian (second derivative) of the image\n- Higher is better'],
        'Tenengrad Method': [blur_metrics[1], m_srn_metrics[1], b_srn_metrics[1], deepdeblur_metrics[1], deblurgan_metrics[1], '- Calculates the gradient magnitude using Sobel operators\n- Higher is better'],
        'Perceptual Sharpness Index (PSI)': [blur_metrics[2], m_srn_metrics[2], b_srn_metrics[2], deepdeblur_metrics[2], deblurgan_metrics[2], '- Combines edge detection and contrast measures\n- Higher is better'],
        'Entropy-Based Method': [blur_metrics[3], m_srn_metrics[3], b_srn_metrics[3], deepdeblur_metrics[3], deblurgan_metrics[3], '- Measures the amount of information in the image using entropy\n- Higher is better'],
        'Wavelet Transform Based Method': [blur_metrics[4], m_srn_metrics[4], b_srn_metrics[4], deepdeblur_metrics[4], deblurgan_metrics[4], '- Decomposes the image into different frequency bands\n- Sharpness is assessed based on the high-frequency components']
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Image')  # Assuming 'Image' column contains 'Blur' and 'Sharp' values
    df = df.transpose()  # transposes the dataframe, effectively flipping it vertically
    df['Metrics'] = df['Metrics'].apply(lambda x: x.replace('\n', '<br>') if isinstance(x, str) else x)
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)


def visual_metrics(blur, deblurred, name_1, name_2):
    st.write('1. Histogram of Image Gradients: Plots the distribution of gradient magnitudes. A wider distribution indicates a sharper image with more edges.')
    plot_gradient_histogram_overlay(blur, deblurred, label1=name_1, label2=name_2)
    st.write('2. Intensity Histogram: Plots the distribution of pixel intensity values. Provides an overview of the image’s contrast and brightness, indirectly indicating sharpness.')
    plot_intensity_histogram_overlay(blur, deblurred, label1=name_1, label2=name_2)
    st.write('3. Edge Map Visualization: Visualizes edges detected in the image using edge detection algorithms like Canny. More and clearer edges suggest a sharper image.')
    plot_edge_map(blur, deblurred, label1=name_1, label2=name_2)
    # st.write('Histogram Equalization: Improves image contrast, making edges and details more visible. Helps visually assess the sharpness by enhancing the image features.')
    # plot_histogram_equalization(deblurred)
