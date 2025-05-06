import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from data import train_data, test_data

# Veriyi Hazırlama
word_to_index = {}
index = 0
encoded_sentences = []
labels = []

# Train veri seti için tokenleştirme ve indeksleme
for sentence, label in train_data.items():
    tokens = sentence.lower().split()
    encoded_sentence = []
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = index
            index += 1
        encoded_sentence.append(word_to_index[token])
    encoded_sentences.append(encoded_sentence)
    labels.append(int(label))

# Verileri padding ile eşitleme (uzunluklarını standart hale getirme)
maxlen = max([len(sentence) for sentence in encoded_sentences])
padded_sentences = pad_sequences(encoded_sentences, maxlen=maxlen, padding='post')

# Kategorik hale getirme (etiketleri one-hot encode)
labels = to_categorical(labels, num_classes=2)

# Modeli Oluşturma
model = Sequential()
model.add(Embedding(input_dim=len(word_to_index), output_dim=64, input_length=maxlen))
model.add(SimpleRNN(64, activation='tanh'))
model.add(Dense(2, activation='softmax'))

# Modeli Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli Eğitme
history = model.fit(padded_sentences, labels, epochs=10, batch_size=32, verbose=1)

# Test Verisini Hazırlama
encoded_test_sentences = []
test_labels = []

# Test veri seti için tokenleştirme
for sentence, label in test_data.items():
    tokens = sentence.lower().split()
    encoded_sentence = [word_to_index.get(token, 0) for token in tokens]
    encoded_test_sentences.append(encoded_sentence)
    test_labels.append(int(label))

# Test verisini padding ile eşitleme
padded_test_sentences = pad_sequences(encoded_test_sentences, maxlen=maxlen, padding='post')

# Test Etme ve Doğruluğu Yazdırma
test_labels = to_categorical(test_labels, num_classes=2)
loss, accuracy = model.evaluate(padded_test_sentences, test_labels)

print(f"Test Accuracy: {accuracy:.4f}")

# Modelin Eğitim Sürecinin Grafikleri
# Accuracy ve Loss grafikleri
plt.figure(figsize=(12, 6))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Test verisi üzerinde tahmin yapma
predictions = model.predict(padded_test_sentences)
predicted_labels = np.argmax(predictions, axis=1)

# Karmaşıklık Matrisi ve Sınıflandırma Raporu
cm = confusion_matrix(np.argmax(test_labels, axis=1), predicted_labels)
print("Confusion Matrix:")
print(cm)

# Sınıflandırma Raporu (Precision, Recall, F1-Score)
cr = classification_report(np.argmax(test_labels, axis=1), predicted_labels, target_names=["Negative", "Positive"])
print("Classification Report:")
print(cr)
