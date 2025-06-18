import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.cm as cm

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🩺 Skin Disease Classifier with Grad-CAM")
st.write("Upload a dermatoscopic image to classify skin disease and visualize model focus with Grad-CAM.")

# Load model
try:
    model = tf.keras.models.load_model("skin_model.h5")
except:
    st.error("❌ Could not load model. Please make sure 'skin_model.h5' exists in this folder.")
    st.stop()

IMG_SIZE = 64
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv3"):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), int(pred_index), predictions.numpy()
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None, None, None

# Image upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    model = tf.keras.models.load_model("skin_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Preprocess image
    img = np.array(image).astype("float32")  # ✅ Fix dtype
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    heatmap, pred_class, probs = make_gradcam_heatmap(img_input, model)
    print("PROBABILITIES:", probs)
    if heatmap is not None:
        # Create Grad-CAM output
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cm.jet(heatmap)[:, :, :3] * 255
        superimposed_img = heatmap_colored * 0.4 + img_resized
        superimposed_img = np.uint8(superimposed_img)

        st.subheader(f"Prediction: {CLASS_NAMES[pred_class]} ({probs[0][pred_class]*100:.2f}%)")
        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)
    else:
        st.warning("⚠️ Grad-CAM could not be generated.")
