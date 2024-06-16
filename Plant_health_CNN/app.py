import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def model_prediction(test_img):
    cnn = tf.keras.models.load_model('Disease_trained_Data.keras')
    image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # 단일 이미지를 배치로 변환합니다.
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)  # 예측된 클래스의 인덱스를 찾습니다. 확률이 가장 높은 클래스의 인덱스를 반환합니다.
    return result_index

st.header("식물 질병 분석 프로그램")
test_img = st.file_uploader("Choose an Image : ")

if st.button("Show Image"):
    if test_img is not None:
        st.image(test_img, use_column_width=True)
    else:
        st.warning("이미지를 선택해주세요.")

if st.button("Prediction"):
    if test_img is not None:
        result_index = model_prediction(test_img)
        
        validation_set = tf.keras.utils.image_dataset_from_directory(
            'valid',
            labels="inferred",
            label_mode="categorical",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(128, 128),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        
        class_name = validation_set.class_names
        st.success("모델 예측 결과 : {}".format(class_name[result_index]))
    else:
        st.warning("이미지를 선택해주세요.")
