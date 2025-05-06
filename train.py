from rnn import RNN
from data_loader import X_train, y_train, X_test, y_test, word_to_ix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
# Model
rnn = RNN(input_size=len(word_to_ix), hidden_size=16, output_size=2)

# Eğitim
losses = []
for epoch in range(20):
    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in perm]
    y_train = [y_train[i] for i in perm]
    total_loss = 0
    for x, y in zip(X_train, y_train):
        scores, hs = rnn.forward(x)
        l = rnn.loss(scores, y)
        total_loss += l
        rnn.backward(x, hs, y)
    losses.append(total_loss)
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Test
y_preds = [rnn.predict(x) for x in X_test]
cm = confusion_matrix(y_test, y_preds)
Disp = ConfusionMatrixDisplay(cm, display_labels=["Neg","Pos"])
Disp.plot(); plt.title("Confusion Matrix"); plt.show()

# Karışıklık matrisi
y_true = y_test
cm = confusion_matrix(y_true, y_preds)
Disp = ConfusionMatrixDisplay(cm, display_labels=["Neg","Pos"])
Disp.plot(); plt.title("Confusion Matrix"); plt.show()


# Diğer metrikler
acc = accuracy_score(y_true, y_preds)
prec = precision_score(y_true, y_preds)
rec = recall_score(y_true, y_preds)
f1 = f1_score(y_true, y_preds)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}\n")

# Sınıflandırma raporu
print("Classification Report:")
print(classification_report(y_true, y_preds, target_names=["Negative","Positive"]))

# Kayıp grafiği
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# cm = confusion_matrix(y_test, y_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# plt.plot(losses)
# plt.title("Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()
