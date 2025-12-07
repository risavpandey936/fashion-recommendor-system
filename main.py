import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pandas as pd

# ----------------- Load data & model -----------------

# image embeddings + filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

filenames_raw = pickle.load(open('filenames.pkl', 'rb'))
# IMPORTANT: normalize all paths to repo's images/ folder
filenames = [os.path.join("images", os.path.basename(p)) for p in filenames_raw]

# styles metadata
styles = pd.read_csv("styles_cleaned.csv")
styles.set_index("id", inplace=True)

# ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# ----------------- Helper functions -----------------

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def get_id_from_path(path):
    base = os.path.basename(path)      # '28543.jpg'
    id_str = os.path.splitext(base)[0] # '28543'
    return int(id_str)

def show_item_with_metadata(col, img_path):
    # SAFETY: avoid crash if file not present on server
    if not os.path.exists(img_path):
        col.write(f"Image not found on server: {img_path}")
        return

    prod_id = get_id_from_path(img_path)

    if prod_id in styles.index:
        row = styles.loc[prod_id]
        col.image(img_path)

        col.markdown(f"**ID:** {prod_id}")
        col.markdown(f"**Name:** {row['productDisplayName']}")
        col.markdown(f"**Gender:** {row['gender']}")
        col.markdown(f"**Category:** {row['masterCategory']} / {row['subCategory']} / {row['articleType']}")
        col.markdown(f"**Color:** {row['baseColour']}")
        year_text = int(row['year']) if not pd.isna(row['year']) else 'NA'
        col.markdown(f"**Season:** {row['season']}  |  **Year:** {year_text}")
        col.markdown(f"**Usage:** {row['usage']}")
    else:
        col.image(img_path)
        col.write("No metadata found for this item.")

# ----------------- Streamlit UI -----------------
st.write("Sample filename[0]:", filenames[0])
st.write("Exists on server?", os.path.exists(filenames[0]))
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image")

        upload_path = os.path.join("uploads", uploaded_file.name)
        features = feature_extraction(upload_path, model)
        indices = recommend(features, feature_list)

        cols = st.columns(5)
        st.write("First recommended index:", int(indices[0][0]))
        st.write("Path for first recommended image:", filenames[indices[0][0]])
        st.write("Exists?", os.path.exists(filenames[indices[0][0]]))

        for rank, col in enumerate(cols):
            img_path = filenames[indices[0][rank]]
            show_item_with_metadata(col, img_path)
    else:
        st.header("Some error occurred in file upload")





