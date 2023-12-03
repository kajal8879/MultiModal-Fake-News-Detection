import numpy as np
from tensorflow import keras
from transformers import BertTokenizer, BertModel
import torch
import cv2
import requests

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
    
def predict_fake_news():
    # Load the pre-trained model
    loaded_model = keras.models.load_model('fake_news_detection_model.keras')

    # User input
    author = input("Enter the author: ")
    clean_title = input("Enter the clean title : ")
    domain =  input("Enter the domain:")
    image_url = input("Enter the image URL: ")

    # Tokenize and encode text

    text_embedding = encode_text(author, clean_title, domain)

    # Vectorize image from URL
    image_embedding = vectorize_image_from_url(image_url, target_size=(224, 224))

    if text_embedding is not None and image_embedding is not None:
        # Reshape for model input
        text_embedding = text_embedding.reshape(1, -1)
        image_embedding = image_embedding.reshape(1, 224, 224, 3)

        # Make predictions
        predictions = loaded_model.predict([text_embedding, image_embedding])

        # Assuming it's a binary classification (sigmoid activation in the output layer)
        binary_prediction = (predictions > 0.5).astype(int)

        # Display the result
        print("Prediction:", "Fake" if binary_prediction[0][0] == 1 else "Not Fake")

# Call the function to predict fake news based on user input
predict_fake_news()
