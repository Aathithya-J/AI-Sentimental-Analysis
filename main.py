import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = {
    "text": [
        "I am so happy today!",
        "This is the worst day ever.",
        "I'm feeling very sad.",
        "Wow, I love this!",
        "I'm really angry with you.",
        "I'm scared of what's coming.",
        "That was a fantastic experience!",
        "I'm anxious about the test.",
        "I'm so proud of myself.",
        "This makes me furious!"
    ],
    "emotion": [
        "happy", "angry", "sad", "happy", "angry",
        "fear", "happy", "fear", "happy", "angry"
    ]
}

df = pd.DataFrame(data)

tokenizer = preprocessing.text.Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

padded_sequences = preprocessing.sequence.pad_sequences(sequences, padding='post')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['emotion'])

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(6, activation='softmax')  # classes: happy, angry, sad, fear, etc.
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

a=input("What is the sentence: ")
test_sentences=[]
test_sentences.append(a)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = preprocessing.sequence.pad_sequences(test_sequences, maxlen=padded_sequences.shape[1], padding='post')

predictions = model.predict(test_padded)
predicted_labels = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

for sentence, emotion in zip(test_sentences, predicted_labels):
    print(f"Text: '{sentence}' -> Predicted Emotion: {emotion}")
