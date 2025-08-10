import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from PIL import Image, ImageOps
import cv2
import gdown
import os
import traceback

# --- تعریف توابع سفارشی مدل ---
@register_keras_serializable()
def rescale_gap(inputs):
    gap_feat, gap_attn = inputs
    return gap_feat / (gap_attn + 1e-7)

# اگر توابع سفارشی دیگر دارید اینجا اضافه کنید
custom_objects = {
    'rescale_gap': rescale_gap,
}

# --- دانلود مدل ---
try:
    file_id = "1HqyEY_5PIocLmJ2MzidLPltpA88An8L7"  # شناسه فایل گوگل درایو مدل شما
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    model_path = "model.h5"

    if not os.path.exists(model_path):
        
        gdown.download(url, model_path, quiet=False)
    else:
        st.success("برنامه در حال اجراست")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
       
except Exception as e:
    st.error("❌ خطا در دانلود مدل:")
    st.text(type(e).__name__ + ": " + str(e))
    st.text(traceback.format_exc())

# --- بارگذاری مدل ---
model = None
try:
    
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
except Exception as e:
    st.error("❌ خطا در بارگذاری مدل:")
    st.text(type(e).__name__ + ": " + str(e))
    st.text(traceback.format_exc())

# --- تابع پیش‌بینی ---
def import_and_predict(image_data, model):
    try:
        size = (200, 200)
        image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
        image = image.convert('RGB')
        image_array = np.asarray(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        return prediction, image_array, image
    except Exception as e:
        st.error("❌ خطا در پیش‌بینی تصویر:")
        st.text(type(e).__name__ + ": " + str(e))
        st.text(traceback.format_exc())
        return None, None, None

# --- Grad-CAM ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            # ابعاد اضافی را حذف کن (مثلا از (1,1,6) به (1,6))
            predictions = tf.squeeze(predictions)  # حالا shape احتمالا (6,) یا (batch_size, classes)

            if pred_index is None:
                pred_index_tensor = tf.argmax(predictions, axis=-1)
                pred_index = int(pred_index_tensor.numpy())

            # اطمینان از اینکه predictions بعد کافی دارد
            class_channel = predictions[pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        return heatmap

    except Exception as e:
        st.error("❌ خطا در ساخت Grad-CAM:")
        st.text(type(e).__name__ + ": " + str(e))
        st.text(traceback.format_exc())
        return None



# --- ترکیب heatmap با تصویر ---
def overlay_heatmap(img, heatmap, alpha=0.4):
    try:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
        return overlay
    except Exception as e:
        st.error("❌ خطا در ترکیب heatmap با تصویر:")
        st.text(type(e).__name__ + ": " + str(e))
        st.text(traceback.format_exc())
        return None

# --- CSS ---
st.markdown("""
    <style>
        .title {font-size: 36px; font-weight: bold; text-align: center; color: #1f77b4;}
        .subtitle {font-size: 22px; font-weight: bold; text-align: center; margin-bottom: 30px; color: #333;}
        .section-title {font-size: 20px; font-weight: bold; color: #444; margin-top: 10px;}
        .result {font-size: 22px; font-weight: bold; color: green;}
        .label-text {font-size: 16px; font-weight: bold; color: #000;}
    </style>
""", unsafe_allow_html=True)

# --- عنوان ---
st.markdown('<div class="title">🔍 سامانه تشخیص عیوب سطح فولاد</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">مدل مبتنی بر VGG16 با مکانیزم توجه (Grad-CAM)</div>', unsafe_allow_html=True)

# --- آپلود تصویر ---
file = st.file_uploader("📂 لطفاً تصویر عیب سطح فولاد را بارگذاری کنید", type=["jpg", "jpeg", "png"])

if file is not None:
    if model is not None:
        prediction, img_array, resized_image = import_and_predict(Image.open(file), model)

        if prediction is not None:
            try:
                class_labels = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled-in_scale', 'Scratches']
                pred_index = np.argmax(prediction)
                pred_label = class_labels[pred_index]
                confidence = prediction[0][pred_index]

                st.markdown(f"<div class='result'>✅ نتیجه تشخیص: {pred_label} ({confidence*100:.2f}%)</div>", unsafe_allow_html=True)

                col1, _, col2, _, col3 = st.columns([1.5, 0.7, 1.5, 0.7, 1.7])

                with col1:
                    st.markdown("<div class='section-title'>📷 تصویر ورودی</div>", unsafe_allow_html=True)
                    st.image(resized_image, width=250)

                with col2:
                    st.markdown("<div class='section-title'>🔥 نقشه توجه (Attention Map)</div>", unsafe_allow_html=True)
                    heatmap = make_gradcam_heatmap(img_array, model)
                    if heatmap is not None:
                        original = np.uint8(255 * img_array[0])
                        attention = overlay_heatmap(original, heatmap)
                        if attention is not None:
                            st.image(attention, width=250)

                with col3:
                    st.markdown("<div class='section-title'>📊 احتمال هر کلاس</div>", unsafe_allow_html=True)
                    for i, label in enumerate(class_labels):
                        st.markdown(f"<div class='label-text'>{label}: {prediction[0][i]:.4f}</div>", unsafe_allow_html=True)
                        st.progress(float(prediction[0][i]))
            except Exception as e:
                st.error("❌ خطا در نمایش نتایج:")
                st.text(type(e).__name__ + ": " + str(e))
                st.text(traceback.format_exc())
    else:
        st.error("❌ مدل بارگذاری نشده است؛ پیش‌بینی ممکن نیست.")
else:
    st.info("📎 لطفاً یک تصویر بارگذاری کنید.")









