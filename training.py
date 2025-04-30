import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def load_intents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading intents: {e}")
        return None

def preprocess_data(intents):
    words, labels, docs_x, docs_y = [], [], [], []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = word_tokenize(pattern.lower())
            word_list = [lemmatizer.lemmatize(word) for word in word_list]
            words.extend(word_list)
            docs_x.append(word_list)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = sorted(set(words))
    labels = sorted(set(labels))
    return words, labels, docs_x, docs_y

def tokenize_and_pad(docs_x, max_len=None):
    tokenizer = Tokenizer(oov_token="<OOV>")
    texts = [" ".join(x) for x in docs_x]
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
    return tokenizer, padded_sequences, max_len

def encode_labels(docs_y, labels):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(docs_y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(labels))
    return encoder, y_categorical

def build_model(input_dim, max_len, num_classes):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=128),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128)),
        Dropout(0.5),
        LayerNormalization(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
                  metrics=["accuracy"])
    return model

def save_artifacts(tokenizer, encoder, tokenizer_path="tokenizer.pkl", encoder_path="encoder.pkl"):
    try:
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        print("Tokenizer and encoder saved successfully.")
    except Exception as e:
        print(f"Error saving artifacts: {e}")

def load_artifacts(tokenizer_path="tokenizer.pkl", encoder_path="encoder.pkl"):
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        return tokenizer, encoder
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None

def predict_class(sentence, tokenizer, encoder, model, max_len):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    seq = tokenizer.texts_to_sequences([" ".join(tokens)])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    prediction = model.predict(padded, verbose=0)[0]
    tag = encoder.inverse_transform([np.argmax(prediction)])
    return tag[0], np.max(prediction)

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_accuracy.png")
    plt.show()

def main():
    intents = load_intents("intents.json")
    if not intents:
        return

    words, labels, docs_x, docs_y = preprocess_data(intents)
    tokenizer, X, max_len = tokenize_and_pad(docs_x)
    encoder, y = encode_labels(docs_y, labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    save_artifacts(tokenizer, encoder)

    model = build_model(input_dim=len(tokenizer.word_index)+1, max_len=max_len, num_classes=len(labels))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
    ]

    try:
        history = model.fit(X_train, y_train, epochs=100, batch_size=128,
                            validation_data=(X_val, y_val), callbacks=callbacks)
        print("Training the model...")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    try:
        model.save("chatbot_model.h5")
        tf.keras.models.save_model(model, "chatbot_model.keras")
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {acc:.4f}")
    model.summary()
    plot_history(history)
    tag, conf = predict_class("hello there!", tokenizer, encoder, model, max_len)
    print(f"Prediction: {tag} (confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
