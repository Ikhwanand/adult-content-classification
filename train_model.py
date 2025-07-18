import os 
import xml.etree.ElementTree as ET 
from PIL import Image 
import numpy as np 
# import matplotlib.pyplot as plt 
# import pandas as pd 
# from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import cv2
from tensorflow.keras.utils import Sequence


def process_data(folder_path):
    data = []
    labels = []
    image_info = []
    
    # Get all image files first
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    for img_filename in image_files:
        try:
            # Process Image 
            img_path = os.path.join(folder_path, img_filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Find corresponding XML file
            xml_filename = img_filename.replace('.jpg', '.xml')
            xml_path = os.path.join(folder_path, xml_filename)
            
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image info
                width = int(root.find("size/width").text)
                height = int(root.find('size/height').text)
                
                # Get all objects in the image
                objects = root.findall('object')
                image_labels = []
                
                for obj in objects:
                    label = obj.find('name').text
                    image_labels.append(label)
                
                # Store data
                data.append(img_array)
                labels.append(image_labels)
                image_info.append({
                    'filename': img_filename,
                    'width': width,
                    'height': height,
                    'num_objects': len(objects),
                    'labels': image_labels
                })
                
            else:
                print(f"XML file not found for {img_filename}")
        except Exception as e: 
            print(f"Error processing {img_filename}: {str(e)}")
            
    return data, labels, image_info


# Preprocessing function
def preprocess_images(data, target_size=(224, 224)):
    """
    Preprocess images for model training
    """
    processed_images = []
    
    for img_array in data:
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        processed_images.append(img_normalized)
    
    return np.array(processed_images)


# Create binary labels (1 for porn, 0 for non-porn)
def create_binary_labels(labels):
    """
    Convert labels to binary classification
    """
    binary_labels = []
    
    for label_list in labels: 
        # If image has any 'porn' label, mark as 1, otherwise 0
        if len(label_list) > 0 and 'porn' in label_list:
            binary_labels.append(1)
        else: 
            binary_labels.append(0)
    return np.array(binary_labels)


def create_cnn_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model 



def create_transfer_learning_model(input_shape=(224, 224, 3)):
    # Load pre-trained MobileNetV2 Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False 
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model 



def evaluate_model_performance(model, X_test, y_test, X_train, y_train, X_valid, y_valid):
    """Comprehensive model evaluation without plots"""
    print('='*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Make predictions on all datasets
    print("\nMaking predictions...")
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_pred_binary = (y_train_pred > 0.5).astype(int).flatten()
    
    # Validation set predictions
    y_valid_pred = model.predict(X_valid)
    y_valid_pred_binary = (y_valid_pred > 0.5).astype(int).flatten()
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    y_test_pred_binary = (y_test_pred > 0.5).astype(int).flatten()
    
    # Calculate metrics for each dataset
    datasets = {
        'Training': (y_train, y_train_pred_binary, y_train_pred),
        'Validation': (y_valid, y_valid_pred_binary, y_valid_pred),
        'Test': (y_test, y_test_pred_binary, y_test_pred)
    }
    
    for dataset_name, (y_true, y_pred_binary, y_pred_prob) in datasets.items():
        print(f"\n{'-'*20} {dataset_name.upper()} SET METRICS {'-'*20}")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, average='binary')
        recall = recall_score(y_true, y_pred_binary, average='binary')
        f1 = f1_score(y_true, y_pred_binary, average='binary')
        
        try:
            auc_score = roc_auc_score(y_true, y_pred_prob.flatten())
        except: 
            auc_score = "N/A"
            
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"AUC-ROC: {auc_score}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 0    1")
        print(f"Actual    0    {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"          1    {cm[1,0]:4d} {cm[1,1]:4d}")
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 
        
        print("\nAdditional Metrics:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred_binary,
                                    target_names=['Non-Porn', 'Porn'],
                                    digits=4))
        
        
    # Model Comparison Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Training':<12} {'Validation':<12} {'Test':<12}")
    print(f"{'-'*60}")
    
    train_acc = accuracy_score(y_train, y_train_pred_binary)
    valid_acc = accuracy_score(y_valid, y_valid_pred_binary)
    test_acc = accuracy_score(y_test, y_test_pred_binary)
    
    train_prec = precision_score(y_train, y_train_pred_binary)
    valid_prec = precision_score(y_valid, y_valid_pred_binary)
    test_prec = precision_score(y_test, y_test_pred_binary)
    
    train_rec = recall_score(y_train, y_train_pred_binary)
    valid_rec = recall_score(y_valid, y_valid_pred_binary)
    test_rec = recall_score(y_test, y_test_pred_binary)
    
    train_f1 = f1_score(y_train, y_train_pred_binary)
    valid_f1 = f1_score(y_valid, y_valid_pred_binary)
    test_f1 = f1_score(y_test, y_test_pred_binary)
    
    
    print(f"{'Accuracy':<12} {train_acc:<12.4f} {valid_acc:<12.4f} {test_acc:<12.4f}")
    print(f"{'Precision':<12} {train_prec:<12.4f} {valid_prec:<12.4f} {test_prec:<12.4f}")
    print(f"{'Recall':<12} {train_rec:<12.4f} {valid_rec:<12.4f} {test_rec:<12.4f}")
    print(f"{'F1-Score':<12} {train_f1:<12.4f} {valid_f1:<12.4f} {test_f1:<12.4f}")
    
    # Overfitting analysis
    print(f"\n{'='*60}")
    print("OVERFITTING ANALYSIS")
    print(f"{'='*60}")
    
    train_val_acc_diff = train_acc - valid_acc
    train_test_acc_diff = train_acc - test_acc
    
    print(f"Training vs Validation Accuracy Difference: {train_val_acc_diff:.4f}")
    print(f"Training vs Test Accuracy Difference: {train_test_acc_diff:.4f}")
    
    if train_val_acc_diff > 0.1:
        print("⚠️  WARNING: Possible overfitting detected (>10% difference)")
    elif train_val_acc_diff > 0.05:
        print("⚠️  CAUTION: Mild overfitting detected (>5% difference)")
    else:
        print("✅ Good generalization performance")
    
    
    # Model predictions distributions
    print(f"\n{'='*60}")
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    for dataset_name, (y_true, y_pred_binary, y_pred_prob) in datasets.items():
        print(f"\n{dataset_name} Set:")
        print(f" Actual Positives: {np.sum(y_true)} ({np.mean(y_true)*100:.2f}%)")
        print(f" Predicted Positives: {np.sum(y_pred_binary)} ({np.mean(y_pred_binary)*100:.2f}%)")
        print(f" Average Confidence: {np.mean(y_pred_prob):.4f}")
        print(f" Min Confidence: {np.min(y_pred_prob):.4f}")
        print(f" Max Confidence: {np.max(y_pred_prob):.4f}")
        
        


def analyze_prediction_confidence(model, X_test, y_test, threshold=0.5):
    """Analyze prediction confidence levels"""
    print(f"\n{'='*60}")
    print("PREDICTION CONFIDENCE ANALYSIS")
    print(f"{'='*60}")
    
    y_pred_prob = model.predict(X_test).flatten()
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    
    # Confidence levels
    high_confidence = np.sum((y_pred_prob > 0.8) | (y_pred_prob < 0.2))
    medium_confidence = np.sum((y_pred_prob >= 0.2) & (y_pred_prob <= 0.8))
    
    print(f"High Confidence Predictions (>0.8 or <0.2): {high_confidence} ({high_confidence/len(y_test)*100:.2f}%)")
    print(f"Medium Confidence Predictions (0.2-0.8): {medium_confidence} ({medium_confidence/len(y_test)*100:.2f}%)")
    
    # Correct predictions by confidence level
    correct_predictions = (y_pred_binary == y_test)
    high_conf_mask = (y_pred_prob > 0.8) | (y_pred_prob < 0.2)
    
    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = np.mean(correct_predictions[high_conf_mask])
        print(f"Accuracy on High Confidence Predictions: {high_conf_accuracy:.4f} ({high_conf_accuracy*100:.2f}%)")
        
    # Threshold analysis
    print("\nThreshold Analysis:")
    threshold = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*50}")
    
    for thresh in threshold:
        y_pred_thresh = (y_pred_prob > thresh).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        print(f"{thresh:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def model_pipeline(type_model="custom", model_saved_name="best_model.h5", epochs=100, X_train=None, y_train=None, X_valid=None, y_valid=None):
    """Model training pipeline"""
    print(f"\n{'='*60}")
    print("MODEL TRAINING PIPELINE")
    print(f"{'='*60}")
    
    model = create_cnn_model() if type_model == "custom" else create_transfer_learning_model()
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    model.summary()
    
    return model 


    

def main():
    train_data, train_labels, train_info = process_data('./data/train')
    test_data, test_labels, test_info = process_data('./data/test')
    valid_data, valid_labels, valid_info = process_data('./data/valid')
    
    # Preprocess all datasets
    print('Preprocessing images...')
    X_train = preprocess_images(train_data)
    y_train = create_binary_labels(train_labels)

    X_test = preprocess_images(test_data)
    y_test = create_binary_labels(test_labels)

    X_valid = preprocess_images(valid_data)
    y_valid = create_binary_labels(valid_labels)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Validation data shape: {X_valid.shape}")
    
    # Data augmentation for training
    # datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     zoom_range=0.2,
    #     shear_range=0.2,
    #     fill_mode='nearest'
    # )

    # datagen.fit(X_train)
    
    model = model_pipeline(
        type_model="custom"
    )

     # Define callbacks 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "./models/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history = model.fit(
        DataGenerator(X_train, y_train, 32),
        steps_per_epoch=len(X_train) // 32,
        epochs=100,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1
    )
    
    evaluate_model_performance(model, X_test, y_test, X_train, y_train, X_valid, y_valid)

    # Analyze prediction confidence
    analyze_prediction_confidence(model, X_test, y_test)



if __name__ == "__main__":
    main()
