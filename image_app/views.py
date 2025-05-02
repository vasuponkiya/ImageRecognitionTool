import os
import numpy as np
import joblib
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
import cv2
from .forms import ImageUploadForm
from PIL import Image
import keras # might not work if path is deprecated

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/model.pkl')
model = joblib.load(MODEL_PATH)

# Class labels
CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(img_path):
    """Preprocess the uploaded image for model prediction."""
    img = cv2.imread(img_path)  # Read the image
    img_resized = cv2.resize(img, (32, 32))  # Resize to match model input
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 32, 32, 3)
    return img_array

def index(request):
    if request.method == "POST" and "image" in request.FILES:
        file = request.FILES["image"]
        file_name = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
        img_full_path = default_storage.path(file_name)

        # Preprocess and predict
        img = preprocess_image(img_full_path)
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_LABELS[predicted_class_index]

        # Pass image URL to template
        return render(request, "index.html", {"prediction": predicted_class_name, "image_url": "/media/uploads/" + file.name})

    return render(request, "index.html")