import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input

# Config
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']

def plot_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Model Accuracy Progress', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Model Loss Progress', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    # Calculate percentage
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Construct combined labels
    labels = np.asarray([f"{val}\n({pct:.1f}%)" for val, pct in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                cbar=True, annot_kws={"size": 12})
    
    plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def main():
    # Load features and labels
    if not (os.path.exists("X.npy") and os.path.exists("y.npy")):
        print("X.npy and y.npy not found. Please run extract_features.py first.")
        return
        
    X = np.load("X.npy")
    y = np.load("y.npy")
    
    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")
    
    # Split into train and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Scale features
    N_train, n_mfcc, time_steps = X_train.shape
    N_test = X_test.shape[0]
    
    # Flatten for StandardScaler
    X_train_flat = X_train.reshape(N_train, -1)
    X_test_flat = X_test.reshape(N_test, -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Save the scaler for inference script
    joblib.dump(scaler, "scaler.joblib")
    print("StandardScaler saved to scaler.joblib")
    
    # Reshape back to 3D: (N, n_mfcc, time_steps)
    X_train = X_train_scaled.reshape(N_train, n_mfcc, time_steps)
    X_test = X_test_scaled.reshape(N_test, n_mfcc, time_steps)
    
    # Expand dims for CNN input: (N, n_mfcc, time_steps, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Build 2D CNN model
    input_shape = (n_mfcc, time_steps, 1)
    print(f"CNN input shape: {input_shape}")
    
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(4, activation='softmax')
    ])
    
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'speech_emotion_recognition_model.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    print("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot history
    plot_history(history)
    
    # Load best model for evaluation (Checkpoint should have saved the best)
    print("Loading the best saved model for evaluation...")
    best_model = tf.keras.models.load_model('speech_emotion_recognition_model.keras')
    
    # Evaluation
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print("             MODEL EVALUATION")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
