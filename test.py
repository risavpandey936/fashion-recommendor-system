import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors


import os
with open('embeddings.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))  

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

print("Embeddings shape:", feature_list.shape)
print("Examples:", filenames[:3])
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result
neighbors = NearestNeighbors(
    n_neighbors=6,
    algorithm='brute',
    metric='euclidean'
)
neighbors.fit(feature_list)
query_path =  "/home/hp/cristofer-maximilian-AqLIkOzWDAk-unsplash.jpg"

query_feature = extract_features(query_path, model)

distances, indices = neighbors.kneighbors(
    [query_feature]
)

# print("Neighbor indices:", indices[0])
# print("Neighbor distances:", distances[0])
for idx in indices[0]:
    print(filenames[idx])


# --- Optional: show query + top-5 similar images (OpenCV) ---





