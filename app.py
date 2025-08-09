import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import gdown
import os
import traceback

# --- تعریف توابع سفارشی مدل (اگر دارید) ---
def my_custom_lambda(x):
    # تابع نمونه؛ تابع واقعی مدل خود را اینجا قرار دهید
    return tf.nn.relu(x)

custom_objects = {
    'my_custom_lambda': my_custom_lambda,
    # اگر توابع سفارشی دیگری دارید اینجا اضافه کنید
}

# --- دانلود مدل ---
try:
    file_id = "1aGAUVtVOjBgYyCZ3hcj14U05MYFUYEAq"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    model_path = "model.h5"

    if not os.path.exists(model_path):
        st.info("در حال دانلود مدل از Google Drive ...")
        gdown.download(url, model_path, quiet=False)
    else:
        st.success("فایل مدل قبلاً دانلود شده است.")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.write(f"✅ مدل موجود است — اندازه: {size_mb:.2f} MB")
except Exception as e:
    st.error("❌ خطا در دانلود مدل:")
    st.text(type(e).__name__ + ": " + str(e))
    st.text(traceback.format_exc())

# --- بارگذاری مدل ---
model = None
try:
    st.write("✅ TensorFlow نسخه:", tf.__version__)
    st.info("در حال بارگذاری مدل...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    st.success("مدل با موفقیت لود شد.")
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

# --- Grad-CAM اصلاح شده ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    try:
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
        except Exception as e:
            st.error("❌ خطا در ایجاد مدل Grad-CAM:")
            st.text(type(e).__name__ + ": " + str(e))
            st.text(traceback.format_exc())
            return None

        with tf.GradientTape() as tape:
            try:
                conv_outputs, predictions = grad_model(img_array)
            except Exception as e:
                st.error("❌ خطا در اجرای مدل Grad-CAM برای دریافت خروجی‌ها:")
                st.text(type(e).__name__ + ": " + str(e))
                st.text(traceback.format_exc())
                return None

            try:
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0], axis=-1)
                    if isinstance(pred_index, tf.Tensor):
                        pred_index = pred_index.numpy()
                    if isinstance(pred_index, (np.ndarray, list)):
                        pred_index = pred_index.item()
            except Exception as e:
                st.error("❌ خطا در تعیین شاخص پیش‌بینی (pred_index):")
                st.text(type(e).__name__ + ": " + str(e))
                st.text(traceback.format_exc())
                return None

            try:
                class_channel = predictions[0][pred_index]
            except Exception as e:
                st.error("❌ خطا در انتخاب کلاس پیش‌بینی‌شده:")
                st.text(type(e).__name__ + ": " + str(e))
                st.text(traceback.format_exc())
                return None

        try:
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        except Exception as e:
            st.error("❌ خطا در محاسبه گرادیان‌ها:")
            st.text(type(e).__name__ + ": " + str(e))
            st.text(traceback.format_exc())
            return None

        try:
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
        except Exception as e:
            st.error("❌ خطا در ساخت نقشه گرمایی (heatmap):")
            st.text(type(e).__name__ + ": " + str(e))
            st.text(traceback.format_exc())
            return None

        try:
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            return heatmap
        except Exception as e:
            st.error("❌ خطا در نرمال‌سازی و تبدیل نقشه گرمایی:")
            st.text(type(e).__name__ + ": " + str(e))
            st.text(traceback.format_exc())
            return None

    except Exception as e:
        st.error("❌ خطای غیرمنتظره در تابع make_gradcam_heatmap:")
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

# --- آپلود ---
file = st.file_uploader("📂 لطفاً تصویر عیب سطح فولاد را بارگذاری کنید", type=["jpg", "jpeg", "png"])

if file is not None:
    if model is not None:
        prediction, img_array, resized_image = import_and_predict(Image.open(file), model)

        if prediction is not None:
            try:
                class_labels = ['Crazing', 'Patches', 'Inclusion', 'Pitted_surface', 'Rolled-in_scale', 'Scratches']
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







