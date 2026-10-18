import tensorflow as tf
import tensorflow_hub as hub
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# 1. Define Helper Functions
original_image_cache = {}

def preprocess_image(image):
    image = np.array(image)
    # Reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Convert to floats in the [0,1] range
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image

def load_image_from_url(img_url):
    user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
    response = requests.get(img_url, headers=user_agent)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image

def load_image_from_file(file_path):
    # Open the image from your computer
    image = Image.open(file_path)
    
    # AI models expect standard 3-channel RGB images. 
    # If you input a PNG with a transparent background (RGBA), this prevents a crash.
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Send it to your existing preprocessing function
    image = preprocess_image(image)
    return image

def show_image(image, title=''):
    image_size = image.shape[1]
    w = (image_size * 6) // 320
    plt.figure(figsize=(w, w))
    plt.imshow(image[0], aspect='equal')
    plt.axis('off')
    plt.title(title)
    plt.show()

# 2. Select and Load the Model Configuration
model_name = "efficientnetv2-s"
model_handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
image_size = 384

print(f"Selected model: {model_name} : {model_handle}")
print(f"Images will be converted to {image_size}x{image_size}")

# 3. Download ImageNet Labels
labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

with open(downloaded_file) as f:
    labels = f.readlines()
    classes = [l.strip() for l in labels]

print("Model and labels loaded successfully! You are ready to classify an image.")


import urllib.request


#image_url = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG"
#image_tensor = load_image_from_url(image_url)
import tkinter as tk
from tkinter import filedialog

# Briefly open and hide a Tkinter window to use its file dialog
root = tk.Tk()
root.withdraw() 

# Open the Windows file explorer so you can click on your image
print("Please select an image file...")
local_image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# Process the image if you actually selected one (and didn't hit cancel)
if local_image_path:
    image_tensor = load_image_from_file(local_image_path)
   
else:
    print("No file selected. Exiting.")
    exit()
# 2. Resize the image (Most classification models specifically require 224x224 pixels)
image_tensor = tf.image.resize(image_tensor, [224, 224])

# 3. Load a pre-trained ImageNet model from TensorFlow Hub
print("Loading the AI model (this might take a minute on the first run)...")
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
model = hub.load(model_url)

# 4. Feed the image to the model to get its predictions
predictions = model(image_tensor)

# 5. Download the text file containing the 1,000 object names the model knows
labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_path, _ = urllib.request.urlretrieve(labels_url)
with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

# 6. Find the highest scoring prediction and match it to the text label
predicted_index = tf.math.argmax(predictions[0]).numpy()
predicted_label = labels[predicted_index]

print("\n--- RESULT ---")
print(f"The model thinks this image is a: {predicted_label}")