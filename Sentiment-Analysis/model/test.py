# test.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_cleaning import load_and_clean
from sklearn.model_selection import train_test_split

# Load data
df = load_and_clean()
X = np.array(df['cleaned_text'].astype(str))
y = df['label'].astype(int).values

# Same split as training
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Load model
model = load_model('sentiment_model.keras')

# Predict
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Results
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))