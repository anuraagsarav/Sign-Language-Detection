import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Path where dataset is stored
DATA_PATH = 'dataset'

# Define signs (labels)
SIGNS = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]

# Load dataset
X, y = [], []
for sign in SIGNS:
    sign_folder = os.path.join(DATA_PATH, sign)
    for file in os.listdir(sign_folder):
        if file.endswith('.json'):
            file_path = os.path.join(sign_folder, file)
            with open(file_path, 'r') as f:
                landmarks = json.load(f)
                landmark_flattened = [coord for point in landmarks for coord in point.values()]
                X.append(landmark_flattened)
                y.append(sign)

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Test the model
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model_path = 'models/sign_language_model.pkl'
if not os.path.exists('models'):
    os.makedirs('models')
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"Model saved to {model_path}")


# Plot Accuracy (Train vs Test)
def plot_accuracy(train_acc, test_acc):
    plt.figure(figsize=(6, 4))
    labels = ['Train', 'Test']
    accuracies = [train_acc, test_acc]
    plt.bar(labels, accuracies, color=['skyblue', 'salmon'])
    plt.title('Train vs Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()

plot_accuracy(train_accuracy, test_accuracy)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot Confusion Matrix for test set
plot_confusion_matrix(y_test, y_pred_test, labels=SIGNS)
    
# Classification Report (Precision, Recall, F1-score)
def plot_classification_report(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    precision = [report[cls]['precision'] for cls in labels]
    recall = [report[cls]['recall'] for cls in labels]
    f1_score = [report[cls]['f1-score'] for cls in labels]

    x = np.arange(len(labels))  # label locations
    width = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precision, width, label='Precision', color='lightgreen')
    ax.bar(x, recall, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1_score, width, label='F1 Score', color='lightskyblue')

    ax.set_xlabel('Classes')
    ax.set_title('Precision, Recall, and F1 Score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    fig.tight_layout()
    plt.show()

# Plot classification report for test set
plot_classification_report(y_test, y_pred_test, labels=SIGNS)
