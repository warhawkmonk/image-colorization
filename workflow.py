import tensorflow as tf
import keras
import numpy as np
import os
import PIL
import streamlit as st
from PIL import Image
import cv2
import time
from keras.models import load_model
import tensorflow.keras.backend as k
st.markdown("<h1 style='text-align: center;'>Image Clorization</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align; center;'>Built with Tensorflow2 & keras</h3>",unsafe_allow_html=True)
gray=np.load("E:\sdbsdcnbnbsdn dsc\gray_scale.npy")
st.sidebar.title('1. choose from 300 images')
i=st.sidebar.number_input(label='Enter a value:',min_value=1,value=1,step=1)
def batch_prep(gray_img,batch_size=100):
    img=np.zeros((batch_size,224,224,3))
    for kb in range(0,3):
        img[:batch_size,:,:,kb]=gray_img[:batch_size]
    return img
img_in=batch_prep(gray,batch_size=300)
st.sidebar.image(gray[i])
start_analyze_file=st.button('colorize')
if start_analyze_file==True:
    with st.spinner(text='colorizing...'):
        time.sleep(1)
    st.cache(allow_output_mutation=True)
    model=tf.keras.models.load_model("E:\sdbsdcnbnbsdn dsc\model.h")
    prediction=model.predict(img_in)
    st.success('Done!')
    st.image(prediction[i].astype('uint8'),clamp=True)
