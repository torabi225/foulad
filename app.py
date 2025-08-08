import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2

# --- بارگذاری مدل ---
model = tf.keras.models.load_model(
    'G:/project/payan/dr.gandomi/foulad/foulad_model.h5',
    compile=False
)

# --- تابع پیش‌بینی ---
def import_and_predict(image_data, model):
    size = (200, 200)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image_array = np.asarray(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return prediction, image_array, image

# --- Grad-CAM ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

# --- ترکیب heatmap با تصویر اصلی ---
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay

# --- CSS سفارشی برای فونت ---
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
        }
        .subtitle {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            color: #333333;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #444444;
            margin-top: 10px;
        }
        .result {
            font-size: 22px;
            font-weight: bold;
            color: green;
        }
        .label-text {
            font-size: 16px;
            font-weight: bold;
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# --- عنوان اصلی ---
st.markdown('<div class="title">🔍 سامانه تشخیص عیوب سطح فولاد</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">مدل مبتنی بر VGG16 با مکانیزم توجه (Grad-CAM)</div>', unsafe_allow_html=True)

# --- آپلود فایل ---
file = st.file_uploader("📂 لطفاً تصویر عیب سطح فولاد را بارگذاری کنید", type=["jpg", "jpeg", "png"])

if file is not None:
    prediction, img_array, resized_image = import_and_predict(Image.open(file), model)

    class_labels = ['Crazing', 'Patches', 'Inclusion', 'Pitted_surface', 'Rolled-in_scale', 'Scratches']
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = prediction[0][pred_index]

    st.markdown(f"<div class='result'>✅ نتیجه تشخیص: {pred_label} ({confidence*100:.2f}%)</div>", unsafe_allow_html=True)

    # --- 3 ستون + فاصله ---
    col1, spacer1, col2, spacer2, col3 = st.columns([1.5, 0.7, 1.5, 0.7, 1.7])

    with col1:
        st.markdown("<div class='section-title'>📷 تصویر ورودی</div>", unsafe_allow_html=True)
        st.image(resized_image, width=250)

    with col2:
        st.markdown("<div class='section-title'>🔥 نقشه توجه (Attention Map)</div>", unsafe_allow_html=True)
        heatmap = make_gradcam_heatmap(img_array, model)
        original = np.uint8(255 * img_array[0])
        attention = overlay_heatmap(original, heatmap)
        st.image(attention, width=250)

    with col3:
        st.markdown("<div class='section-title'>📊 احتمال هر کلاس</div>", unsafe_allow_html=True)
        for i, label in enumerate(class_labels):
            st.markdown(f"<div class='label-text'>{label}: {prediction[0][i]:.4f}</div>", unsafe_allow_html=True)
            st.progress(float(prediction[0][i]))

else:
    st.info("📎 لطفاً یک تصویر بارگذاری کنید.")
