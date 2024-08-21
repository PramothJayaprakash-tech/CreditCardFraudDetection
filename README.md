# Enhanced Credit Card Fraud Detection Using RNN-LSTM with Attention Mechanism

**Contributors:**  
- Rithika Murali - [University of Windsor, Ontario, Canada](https://www.uwindsor.ca)  
- Pramoth Jayaprakash - [University of Windsor, Ontario, Canada](https://www.uwindsor.ca)  
- Gowtham Baskar - [University of Windsor, Ontario, Canada](https://www.uwindsor.ca)  
- La Li - [University of Windsor, Ontario, Canada](https://www.uwindsor.ca)

## Overview

Credit card fraud in online transactions poses significant security challenges, necessitating sophisticated detection systems. This project introduces a novel fraud detection system targeting online transactions by utilizing Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells, integrated with attention mechanisms to capture temporal dependencies and highlight critical transaction details. This approach significantly enhances detection accuracy, achieving an impressive 99.97% accuracy, thereby reducing false positives and negatives.

## Project Structure

### 1. Introduction

The rise of online shopping has heightened the risk of credit card fraud, leading to significant financial losses. Traditional fraud detection systems, relying on rule-based approaches and manual reviews, are no longer effective. Our project aims to address this issue by introducing an advanced fraud detection system tailored specifically for online transactions.

### 2. Related Work

Our work builds upon previous studies that have implemented various machine learning models for fraud detection, such as Random Forests (RF) with Synthetic Minority Over-sampling Technique (SMOTE) and AllKNN-CatBoost, achieving notable accuracies. However, our model surpasses these with a combination of RNN-LSTM and attention mechanisms.

### 3. Methodology

#### 3.1 Data Collection and Preprocessing

- **Dataset:** Sourced from Kaggle, consisting of 568,630 credit card transactions labeled as fraud or non-fraud.
- **Preprocessing:** The dataset was cleaned, normalized, and split into training and testing sets (80-20 split). The data was reshaped to fit the LSTM model's 3D input requirements, treating each transaction as a single timestep with multiple features.

#### 3.2 Model Architecture

- **Core Model:** The architecture integrates RNNs with LSTM cells and attention mechanisms to focus on significant transaction features.
- **Training:** The model was trained using the Adam optimizer and binary cross-entropy loss, running for ten epochs with a batch size of 64.
- **Evaluation:** Performance metrics like accuracy, precision, recall, and F1-score were calculated, with results visualized through confusion matrices and classification reports.

### 4. Results

- **Accuracy:** Achieved 99.97%, with a minimal number of false positives and false negatives.
- **Training and Validation:** High accuracy with minimal overfitting, as indicated by the training and validation loss plots.
- **Class Distribution:** Balanced class distribution ensured effective model training and reliable fraud detection.

### 5. Limitations

Despite the high accuracy, the model may underperform in real-time scenarios with highly imbalanced datasets or evolving fraud patterns. The computational complexity of deep learning models like RNNs with attention mechanisms can also be a limitation.

### 6. Conclusions and Future Work

Our advanced fraud detection system demonstrates significant improvements over previous works, setting a new benchmark in the field. Future work includes incorporating real-time data, exploring hybrid models, and enhancing the scalability of the system for broader deployment.

## References

- [Credit Card Fraud Detection Using Deep Learning](https://doi.org/10.1109/RAICS51191.2020.9332497)
- [Enhanced Random Forest Classifier for Imbalanced Data](https://doi.org/10.1007/s11042-024-18764-1)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Code :

# Section 1: Import Libraries and Load Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Permute, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/creditcard_2023.csv'
data = pd.read_csv(file_path)

# Check for missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()

# Section 2: Filter Transactions and Prepare Data
# Filter the transactions where is_online is True
online_data = data[data['is_online'] == True]

# Features and target variable
X = online_data.drop(columns=['Class', 'is_online'])
y = online_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Section 3: Define Attention Mechanism
# Attention mechanism
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(input_dim, activation='softmax')(a)
    a = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a])
    return Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)

# Section 4: Build and Compile the Model
# Build the model
inputs = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
lstm_out = LSTM(units=50, return_sequences=True)(inputs)
attention_out = attention_3d_block(lstm_out)
output = Dense(1, activation='sigmoid')(attention_out)

model = Model(inputs=[inputs], outputs=[output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Section 5: Train the Model
# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Section 6: Evaluate the Model
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred = model.predict(X_test_reshaped)
y_pred_class = (y_pred > 0.5).astype(int)

# Visualize the results
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.yticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()

# Classification Report
print(classification_report(y_test, y_pred_class, target_names=['Non-Fraud', 'Fraud']))

# Section 7: Display Results
# Display the results
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_class.flatten()})
fraud_transactions = results[results['Predicted'] == 1]
non_fraud_transactions = results[results['Predicted'] == 0]

print("Fraud Transactions:\n", fraud_transactions)
print("Non-Fraud Transactions:\n", non_fraud_transactions)

# Visualize the balance of the dataset
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()

## Output:

![matrix (2)](https://github.com/user-attachments/assets/25784d55-708c-44f4-a1fb-be7a4b25574a)

![accuracy (2)](https://github.com/user-attachments/assets/8956af66-9a87-493e-b902-e9b0ef8f27cd)

![3D visualization (1)](https://github.com/user-attachments/assets/d9177402-cd1f-48f0-83f2-256e56fecfd8)

![results (3)](https://github.com/user-attachments/assets/ec053b8b-2627-4477-af3d-d2491b0d825e)
