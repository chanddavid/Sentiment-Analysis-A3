# train.py
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_cleaning import load_and_clean
# from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout, Input,Bidirectional, LSTM
from tensorflow.keras.layers import (
    Input,TextVectorization, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load cleaned data
df = load_and_clean()
print(f"Total reviews: {len(df)}")

# 2. Split into features and labels
X = np.array(df['cleaned_text'].astype(str))
y = df['label'].astype(int).values

# 3. Train / validation / test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 4. Define TextVectorization layer
max_tokens = 20000
output_sequence_length = 120

vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=2 
)
print("done defining vectorization layer")

# 5. Adapt the layer to the training data
vectorize_layer.adapt(X_train)

# 6. Build the model using Functional APIinputs = Input(shape=(), dtype=tf.string)
inputs = Input(shape=(), dtype=tf.string)
x = vectorize_layer(inputs)

# Embedding layer (bigger = better understanding)
x = Embedding(max_tokens, 64)(x)

# CNN layer → captures phrases like "not good", "very bad"
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)

# LSTM → understands sequence/context
x = Bidirectional(LSTM(32, return_sequences=True))(x)

# Global pooling (keeps strongest signals)
x = GlobalMaxPooling1D()(x)

# Dense layers
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.6)(x)

x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()
print("done building model")


# 7. Train with early stopping
early_stop = EarlyStopping(patience=2, restore_best_weights=True)
print("starting training...")
# Important: Reshape X_train to (samples, 1) to match the Input shape (1,)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    class_weight={0: 1.0, 1: 1.0},
    verbose=1
)
print("done training")

# 8. Save the model
model.save('sentiment_model.keras')
print("Model saved as 'sentiment_model.keras'")
# 9. Evaluate on test set (also reshape)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}") 