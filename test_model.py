import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 64
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Load the model
model = tf.keras.models.load_model("skin_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load a sample image
img = Image.open("sample.jpg").convert("RGB")
img = np.array(img).astype("float32")
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
img_input = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img_input)
print("Prediction probabilities:", predictions)
print("Predicted class:", CLASS_NAMES[np.argmax(predictions)])
