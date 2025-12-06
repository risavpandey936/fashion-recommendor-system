import streamlit as st
import os
import gdown
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

# Google Drive FILE IDs (replace these with your actual FILE IDs from the Drive links)
EMBEDDINGS_ID = "1uyH8Tw3E1R_qQwJz-UR9U2miZzMMmL_v"  # Extract from your copied link
FILENAMES_ID  = "17oP4b7X9H4S5d1IDicAvifgawkxlVrvg"   # Extract from your copied link

EMBEDDINGS_PATH = "embeddings.pkl"
FILENAMES_PATH  = "filenames.pkl"

def download_from_drive(file_id, output_path):
    """Download file from Google Drive if not already present"""
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        st.write(f"📥 Downloading {output_path} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        st.write(f"✅ {output_path} downloaded successfully!")

# Download pickle files on first run
download_from_drive(EMBEDDINGS_ID, EMBEDDINGS_PATH)
download_from_drive(FILENAMES_ID, FILENAMES_PATH)

# Load embeddings and filenames
with open(EMBEDDINGS_PATH, "rb") as f:
    feature_list = np.array(pickle.load(f))

with open(FILENAMES_PATH, "rb") as f:
    filenames = pickle.load(f)

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Build model
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Fit NearestNeighbors once
neighbors = NearestNeighbors(n_neighbors=6,
                             algorithm='brute',
                             metric='euclidean')
neighbors.fit(feature_list)

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except:
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features):
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path is not None:
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Query image", use_column_width=True)

        st.write("🔍 Finding similar fashion items...")
        features = feature_extraction(saved_path, model)
        indices = recommend(features)

        st.write("### 👗 Top 5 Similar Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_column_width=True)
    else:
        st.header("❌ Some error occurred in file upload")

