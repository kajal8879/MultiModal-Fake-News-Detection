import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/kajal/code library/nlp/multimodal_test_public.tsv', sep='\t')

df=df.head(10000)
labels = df['2_way_label'].values

train_texts, test_texts, train_labels, test_labels = train_test_split(df[['author','clean_title','domain','image_url','score','num_comments','subreddit','upvote_ratio','2_way_label']], labels, test_size=0.2, random_state=42)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def encode_text(author, clean_title, domain):
    text = f"{author} {clean_title} {domain}"
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = bert_model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def vectorize_image_from_url(url, target_size=(224, 224)):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for successful response
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Failed to decode image from URL.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)  # Resize to a consistent size
        image_array = image.astype(np.float32) / 255.0
        tensor_form = image_array

        print(f"Image loaded successfully from URL: {url}")
        return tensor_form

    except Exception as e:
        print(f"Error processing image from URL {url}: {str(e)}")
        return None
train_embeddings = []
X_train_image = []
train_labels=[]
target_size = (224, 224)
for author, clean_title, domain, img_url,lbl in zip(train_texts['author'], train_texts['clean_title'], train_texts['domain'], train_texts['image_url'],train_texts['2_way_label']):
    text_embedding = encode_text(author, clean_title, domain)
    image_embedding = vectorize_image_from_url(img_url, target_size)

    if text_embedding is not None and image_embedding is not None:
        train_embeddings.append(text_embedding)
        X_train_image.append(image_embedding)
        train_labels.append(lbl)

train_embeddings = np.array(train_embeddings)
X_train_image = np.array(X_train_image)
text_input = layers.Input(shape=(train_embeddings.shape[1],))
text_layer = layers.Dense(128, activation='relu')(text_input)
image_input = layers.Input(shape=(224, 224, 3))
image_layer = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
image_layer = layers.MaxPooling2D((2, 2))(image_layer)
image_layer = layers.Flatten()(image_layer)
merged = layers.concatenate([text_layer, image_layer])
output = layers.Dense(1, activation='sigmoid')(merged)
print("-----------")
print(output)
train_labels = np.array(train_labels)
model = keras.Model(inputs=[text_input,  image_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(train_embeddings.shape)
print(X_train_image.shape)
print(train_labels.shape)
# Train the model
model.fit([train_embeddings, X_train_image], train_labels, epochs=5, batch_size=32, validation_split=0.2)
model.save('fake_news_detection_model.keras')