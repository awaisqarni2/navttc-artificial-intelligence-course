################################################################################
# REAL-WORLD NEURAL NETWORK: HEXORA CUSTOMER CHURN PREDICTION
# 
# Problem: Predict which customers will cancel their subscription
# Dataset: Customer features (age, tenure, usage, payment history, etc)
# Goal: Build a neural network to identify at-risk customers
# Business Impact: Retain customers before they churn, save revenue
#
# Topics Covered:
# - Forward pass (input → hidden → output)
# - Activation functions (ReLU, Sigmoid)
# - Cross-entropy loss
# - Backpropagation & gradient descent
# - Dropout regularization
# - Batch normalization
# - Early stopping
# - Model evaluation & interpretation
################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

print("=" * 80)
print("HEXORA: Customer Churn Prediction Neural Network")
print("=" * 80)

################################################################################
# PART 1: GENERATE SYNTHETIC DATASET
# In real life, this comes from your database
################################################################################

print("\n[STEP 1] Generating synthetic customer dataset...")

np.random.seed(42)
tf.random.set_seed(42)

# Create synthetic data: 5000 customers with 10 features
n_customers = 5000

data = {
    'customer_id': np.arange(1, n_customers + 1),
    
    # FEATURES (what we use to predict churn)
    'age': np.random.randint(18, 80, n_customers),
    # Customer age (18-80 years)
    
    'tenure_months': np.random.randint(1, 72, n_customers),
    # How long customer has been with us (1-72 months)
    
    'monthly_charges': np.random.uniform(20, 150, n_customers),
    # How much they pay per month (20000-150000)
    
    'total_charges': np.random.uniform(100, 10000, n_customers),
    # Total amount spent with us (100000-10000k)
    
    'support_tickets': np.random.randint(0, 20, n_customers),
    # Number of support tickets opened (0-20)
    
    'payment_delays': np.random.randint(0, 5, n_customers),
    # Number of late payments (0-5)
    
    'contract_length_months': np.random.choice([1, 12, 24], n_customers),
    # Contract duration (1, 12, or 24 months)
    
    'internet_speed_mbps': np.random.choice([25, 50, 100, 500], n_customers),
    # Internet speed package (25, 50, 100, or 500 Mbps)
    
    'customer_satisfaction': np.random.uniform(1, 5, n_customers),
    # Satisfaction score (1-5 stars)
    
    'tech_adoption_score': np.random.uniform(0, 100, n_customers),
    # How tech-savvy customer is (0-100 score)
}

df = pd.DataFrame(data)

# TARGET: Does customer churn? (1=churn, 0=stay)
# Create realistic churn: more likely if low satisfaction, high delays, low tenure
churn_probability = (
    0.3 +  # Base probability
    0.005 * (5 - df['customer_satisfaction']) +  # Higher if unsatisfied
    0.05 * df['payment_delays'] +  # Higher if late payments
    0.003 * (1 / (df['tenure_months'] + 1)) -  # Higher if new customer
    0.001 * df['customer_satisfaction']  # Lower if satisfied
)

df['churn'] = (np.random.random(n_customers) < churn_probability).astype(int)

print(f"✓ Generated {n_customers} customers")
print(f"✓ Churn rate: {df['churn'].mean()*100:.1f}%")
print(f"\nFirst 5 customers:")
print(df.head())

################################################################################
# PART 2: DATA PREPROCESSING
# Critical: Neural networks need normalized, clean data
################################################################################

print("\n[STEP 2] Preprocessing data...")

# Extract features (X) and target (y)
X = df.drop(['customer_id', 'churn'], axis=1).values
y = df['churn'].values

# Why normalize? Neural networks work better with small numbers
# ReLU activation doesn't blow up, gradients are stable, training is faster
scaler = StandardScaler()
# StandardScaler: (x - mean) / std_dev → transforms to mean=0, std=1
X_scaled = scaler.fit_transform(X)

print(f"✓ Features shape: {X_scaled.shape}")  # (5000, 10) - 5000 customers, 10 features
print(f"✓ Feature means (should be ~0): {X_scaled.mean(axis=0)[:3]}")
print(f"✓ Feature stds (should be ~1): {X_scaled.std(axis=0)[:3]}")

# Split into train/test
# 80% training, 20% testing (never train on test data!)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} customers")
print(f"✓ Test set: {X_test.shape[0]} customers")
print(f"✓ Training churn rate: {y_train.mean()*100:.1f}%")
print(f"✓ Test churn rate: {y_test.mean()*100:.1f}%")

################################################################################
# PART 3: BUILD THE NEURAL NETWORK ARCHITECTURE
# 
# Network structure:
# Input(10) → Dense(64, ReLU) → BatchNorm → Dropout(0.3)
#          → Dense(32, ReLU) → BatchNorm → Dropout(0.3)
#          → Dense(16, ReLU) → BatchNorm → Dropout(0.2)
#          → Dense(1, Sigmoid) → Output(0-1 probability)
#
# Why this structure?
# - 10 input features (what we know about customer)
# - 3 hidden layers with decreasing neurons (64→32→16)
#   Each layer learns more abstract patterns
# - ReLU activation: fast, avoids vanishing gradients, works great in practice
# - BatchNormalization: stabilizes training, allows higher learning rates
# - Dropout: prevents overfitting by forcing network to learn redundant features
# - Sigmoid output: squashes to 0-1 probability for binary classification
################################################################################

print("\n[STEP 3] Building neural network architecture...")

model = keras.Sequential([
    # ===== INPUT LAYER =====
    # Receives 10 features per customer
    
    # ===== HIDDEN LAYER 1 =====
    layers.Dense(
        units=64,  # 64 neurons in this layer
        activation='relu',  # ReLU activation: max(0, z)
        # Why ReLU? Fast computation, avoids vanishing gradients, works empirically
        input_shape=(10,)  # Input: 10 features
    ),
    # Parameters: 10*64 weights + 64 biases = 704
    
    layers.BatchNormalization(),
    # Normalizes outputs of Dense layer
    # Stabilizes gradient flow, allows higher learning_rate
    # Effect: training converges 2-3× faster
    
    layers.Dropout(rate=0.3),
    # During training: randomly disable 30% of neurons
    # Effect: Forces network to learn with partial information → less overfitting
    # During inference (predict): use all neurons
    
    # ===== HIDDEN LAYER 2 =====
    layers.Dense(
        units=32,
        activation='relu'
    ),
    # Parameters: 64*32 weights + 32 biases = 2080
    # Reduces dimensionality: 64 → 32
    
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    
    # ===== HIDDEN LAYER 3 =====
    layers.Dense(
        units=16,
        activation='relu'
    ),
    # Parameters: 32*16 weights + 16 biases = 528
    # Further dimensionality reduction: 32 → 16
    
    layers.BatchNormalization(),
    layers.Dropout(rate=0.2),  # Slightly lower dropout here
    
    # ===== OUTPUT LAYER =====
    layers.Dense(
        units=1,  # Single output neuron
        activation='sigmoid'  # Sigmoid: outputs 0-1 probability
        # Formula: sigmoid(z) = 1 / (1 + e^-z)
        # Interpretation: if output ≥ 0.5 → predict churn (1)
        #                 if output < 0.5 → predict stay (0)
    )
    # Parameters: 16*1 weights + 1 bias = 17
])

# Print network architecture
print("\nNetwork Architecture:")
model.summary()

# Output:
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 64)                704
#  batch_normalization         (None, 64)                256
#  dropout (Dropout)           (None, 64)                0
#  dense_1 (Dense)             (None, 32)                2080
#  batch_normalization_1       (None, 32)                128
#  dropout_1 (Dropout)         (None, 32)                0
#  dense_2 (Dense)             (None, 16)                528
#  batch_normalization_2       (None, 16)                64
#  dropout_2 (Dropout)         (None, 16)                0
#  dense_3 (Dense)             (None, 1)                 17
# =================================================================
# Total params: 3,777
# Trainable params: 3,649
# Non-trainable params: 128 (batch norm scale/shift)

################################################################################
# PART 4: COMPILE THE MODEL
# Tell TensorFlow how to train the network
################################################################################

print("\n[STEP 4] Compiling model...")

model.compile(
    # ===== OPTIMIZER =====
    optimizer='adam',
    # Adam (Adaptive Moment Estimation)
    # - Automatically adjusts learning rate per weight
    # - Usually "just works" without tuning
    # - Fast convergence, stable
    # Alternative: keras.optimizers.Adam(learning_rate=0.001)
    
    # ===== LOSS FUNCTION =====
    loss='binary_crossentropy',
    # Binary crossentropy = standard loss for binary classification
    # Formula: L = -(y*log(pred) + (1-y)*log(1-pred))
    # If actual=1 and pred=0.9: loss = -log(0.9) ≈ 0.105 (small penalty)
    # If actual=1 and pred=0.1: loss = -log(0.1) ≈ 2.303 (large penalty)
    # Motivates network to make confident correct predictions
    
    # ===== METRICS =====
    metrics=['accuracy']
    # Track accuracy during training (not used for gradient computation)
    # Accuracy = (correct predictions) / (total predictions)
)

print("✓ Optimizer: Adam")
print("✓ Loss: Binary Crossentropy (standard for binary classification)")
print("✓ Metrics: Accuracy")

################################################################################
# PART 5: DEFINE CALLBACKS
# Automate training process: early stopping, save best model, etc
################################################################################

print("\n[STEP 5] Setting up callbacks...")

# Early Stopping: Stop training if validation loss doesn't improve
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Watch validation loss
    patience=10,  # Stop if no improvement for 10 consecutive epochs
    restore_best_weights=True,  # Revert to best weights (before overfitting)
    verbose=1
)
# Why? Prevents overfitting. Neural nets can memorize training data.
# When validation loss stops improving, training data is overfit.

# Model Checkpoint: Save best model during training
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_churn_model.h5',  # Save to this file
    monitor='val_accuracy',  # Track validation accuracy
    save_best_only=True,  # Only save if this epoch is better than previous
    verbose=1
)
# Why? Insurance policy. If training crashes, we have the best model saved.

# Reduce Learning Rate: If stuck, reduce learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Multiply learning rate by 0.5
    patience=5,  # Wait 5 epochs with no improvement
    min_lr=1e-6,  # Don't go below this
    verbose=1
)
# Why? Fine-tune training. If loss plateaus, smaller steps help escape local minima.

print("✓ Early Stopping: Stop if val_loss doesn't improve for 10 epochs")
print("✓ Model Checkpoint: Save best model to 'best_churn_model.h5'")
print("✓ Learning Rate Reduction: Reduce LR if stuck")

################################################################################
# PART 6: TRAIN THE NEURAL NETWORK
#
# Forward Pass: x → Dense(64, ReLU) → ... → Dense(1, Sigmoid) → output
# Loss Calculation: Compare output to actual label
# Backward Pass (Backpropagation): Chain rule computes gradients ∂Loss/∂weight
# Weight Update: weight = weight - learning_rate * gradient
# Repeat for 100 epochs
################################################################################

print("\n[STEP 6] Training the neural network...")
print("This will take ~30 seconds...\n")

history = model.fit(
    x=X_train,  # Training features (5000, 10)
    y=y_train,  # Training labels (5000,)
    
    epochs=100,
    # Loop through training data 100 times
    # Each epoch: forward pass + backward pass on all training samples
    # Early stopping will stop before 100 if validation loss plateaus
    
    batch_size=32,
    # Process 32 samples at a time
    # 5000 / 32 = 156 batches per epoch
    # Each batch: forward pass + compute loss + backward pass + update weights
    # Smaller batch = noisier but frequent updates
    # Larger batch = smoother but slower updates
    
    validation_split=0.2,
    # Hold back 20% of training data for validation
    # Training set becomes: 4000 training, 1000 validation
    # Validation loss used for early stopping & monitoring overfitting
    
    callbacks=[early_stop, checkpoint, reduce_lr],
    # Run callbacks after each epoch
    
    verbose=1
    # Print progress: Epoch [X/Y], loss, accuracy, val_loss, val_accuracy
)

# Output example:
# Epoch 1/100
# 125/125 [======] - 1s - loss: 0.6821 - accuracy: 0.5432 - val_loss: 0.6234 - val_accuracy: 0.6012
# Epoch 2/100
# 125/125 [======] - 1s - loss: 0.5932 - accuracy: 0.6543 - val_loss: 0.5123 - val_accuracy: 0.6892
# ...
# Epoch 23/100
# 125/125 [======] - 1s - loss: 0.2341 - accuracy: 0.8934 - val_loss: 0.2876 - val_accuracy: 0.8654
# Epoch 24/100
# 125/125 [======] - 1s - loss: 0.2234 - accuracy: 0.8967 - val_loss: 0.2934 - val_accuracy: 0.8612
# (no improvement for 10 epochs, early stopping triggered)

print("\n✓ Training complete!")
print(f"✓ Trained for {len(history.history['loss'])} epochs (early stopping at epoch {len(history.history['loss'])})")

################################################################################
# PART 7: EVALUATE ON TEST SET
# Test set was never seen during training
# True measure of real-world performance
################################################################################

print("\n[STEP 7] Evaluating on test set (unseen data)...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")

# Compare to training accuracy
final_train_accuracy = history.history['accuracy'][-1]
print(f"\n  Training Accuracy:   {final_train_accuracy*100:.2f}%")
print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"  Test Accuracy:       {test_accuracy*100:.2f}%")

# Interpretation:
# - If all three are close (e.g., 85%, 84%, 83%) → good generalization ✓
# - If training >> validation/test (e.g., 95%, 80%, 80%) → overfitting ✗
# - If training ≈ validation << test (rare) → lucky test set

################################################################################
# PART 8: MAKE PREDICTIONS
# Use the trained network on new customers
################################################################################

print("\n[STEP 8] Making predictions on test set...")

# Get probability predictions for all test samples
predictions_prob = model.predict(X_test)  # Shape: (1000, 1)

print(f"✓ Predictions shape: {predictions_prob.shape}")
print(f"✓ Prediction range: [{predictions_prob.min():.4f}, {predictions_prob.max():.4f}]")

# Convert probabilities to binary predictions (churn/stay)
predictions_binary = (predictions_prob >= 0.5).astype(int).flatten()

# Show first 10 predictions
print(f"\nFirst 10 predictions (actual | predicted_prob | predicted_class):")
for i in range(10):
    actual = "CHURN" if y_test[i] == 1 else "STAY"
    pred_class = "CHURN" if predictions_binary[i] == 1 else "STAY"
    print(f"  {i+1}. {actual:5s} | {predictions_prob[i,0]:.4f} | {pred_class:5s}")

################################################################################
# PART 9: DETAILED EVALUATION METRICS
# Accuracy alone is not enough for imbalanced datasets
################################################################################

print("\n[STEP 9] Detailed evaluation metrics...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions_binary)
precision = precision_score(y_test, predictions_binary)
recall = recall_score(y_test, predictions_binary)
f1 = f1_score(y_test, predictions_binary)

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
# What fraction of predictions are correct?
# Good for balanced datasets, not great for imbalanced

print(f"  Precision: {precision*100:.2f}%")
# Of customers we predict will CHURN, how many actually do?
# High precision = few false alarms (low cost of acting on prediction)

print(f"  Recall:    {recall*100:.2f}%")
# Of customers who actually CHURN, how many do we catch?
# High recall = find most churners (low cost of missing them)

print(f"  F1 Score:  {f1:.4f}")
# Harmonic mean of precision & recall (0-1, higher is better)

# Business interpretation:
# Precision = "If we send retention offer, will they be saved?" (cost of false alarms)
# Recall = "How many churners do we catch?" (cost of missing churners)
# Trade-off: offer retention to everyone (high recall, low precision)
#        vs. offer only to high-risk (low recall, high precision)

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions_binary)
print(f"  True Negatives:  {cm[0,0]:4d} (correctly predicted STAY)")
print(f"  False Positives: {cm[0,1]:4d} (predicted CHURN but actually STAY)")
print(f"  False Negatives: {cm[1,0]:4d} (predicted STAY but actually CHURN)")
print(f"  True Positives:  {cm[1,1]:4d} (correctly predicted CHURN)")

print(f"\nClassification Report:")
print(classification_report(y_test, predictions_binary, target_names=['Stay', 'Churn']))

################################################################################
# PART 10: ROC CURVE & AUC
# Visualize trade-off between True Positive Rate and False Positive Rate
################################################################################

print("\n[STEP 10] Computing ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
roc_auc = auc(fpr, tpr)

print(f"✓ AUC Score: {roc_auc:.4f}")
print(f"  AUC = 1.0: Perfect classifier")
print(f"  AUC = 0.5: Random classifier (no better than coin flip)")
print(f"  AUC = {roc_auc:.3f}: {'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Fair'}")

################################################################################
# PART 11: VISUALIZE TRAINING HISTORY
# Plot loss and accuracy over epochs
################################################################################

print("\n[STEP 11] Creating training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training vs Validation Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Loss over Training', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Interpretation:
# - Loss decreases = network is learning
# - Validation loss plateaus = diminishing returns
# - If val_loss increases while train_loss decreases = overfitting

# Plot 2: Training vs Validation Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Accuracy over Training', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=12)
axes[1, 0].set_xlabel('Predicted', fontsize=12)

# Plot 4: ROC Curve
axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})', linewidth=2)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('churn_prediction_analysis.png', dpi=150)
print("✓ Saved visualizations to 'churn_prediction_analysis.png'")
plt.show()

################################################################################
# PART 12: SAVE THE MODEL FOR PRODUCTION
################################################################################

print("\n[STEP 12] Saving trained model for production...")

# Save the entire model (architecture + weights)
model.save('hexora_churn_model.h5')
print("✓ Saved model to 'hexora_churn_model.h5'")

# Save the scaler (needed for preprocessing new data)
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler to 'scaler.pkl'")

################################################################################
# PART 13: LOAD MODEL AND PREDICT ON NEW CUSTOMER
# Simulate production usage
################################################################################

print("\n[STEP 13] Production inference: predict for new customer...")

# Load the saved model
loaded_model = keras.models.load_model('hexora_churn_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# New customer data (must be in same order as training features)
new_customer = pd.DataFrame({
    'age': [35],
    'tenure_months': [6],  # New customer (only 6 months)
    'monthly_charges': [120],
    'total_charges': [720],
    'support_tickets': [8],  # Multiple support tickets
    'payment_delays': [2],  # Some late payments
    'contract_length_months': [1],  # Month-to-month
    'internet_speed_mbps': [50],
    'customer_satisfaction': [2.5],  # Low satisfaction
    'tech_adoption_score': [30],
})

# Preprocess new customer (CRITICAL: use loaded scaler, not fit new one!)
new_customer_scaled = loaded_scaler.transform(new_customer)

# Make prediction
churn_probability = loaded_model.predict(new_customer_scaled, verbose=0)[0, 0]
churn_prediction = "WILL CHURN" if churn_probability >= 0.5 else "WILL STAY"

print(f"\nNew Customer Profile:")
print(f"  Age: {new_customer['age'].values[0]} years old")
print(f"  Tenure: {new_customer['tenure_months'].values[0]} months")
print(f"  Monthly Charge: ${new_customer['monthly_charges'].values[0]:.2f}")
print(f"  Satisfaction: {new_customer['customer_satisfaction'].values[0]}/5")
print(f"  Support Tickets: {new_customer['support_tickets'].values[0]}")

print(f"\nPrediction:")
print(f"  Churn Probability: {churn_probability*100:.2f}%")
print(f"  Decision: {churn_prediction}")

if churn_probability >= 0.75:
    print(f"\n  ⚠️ HIGH RISK! Send retention offer ASAP")
    print(f"     Suggested action: $50 discount + free premium support for 3 months")
elif churn_probability >= 0.5:
    print(f"\n  ⚠️ MODERATE RISK. Monitor closely")
    print(f"     Suggested action: Proactive support outreach")
else:
    print(f"\n  ✓ LOW RISK. Maintain relationship normally")

################################################################################
# PART 14: BUSINESS IMPACT & ROI
################################################################################

print("\n" + "="*80)
print("BUSINESS IMPACT ANALYSIS")
print("="*80)

# Assumptions
customer_lifetime_value = 1500  # $1500 average CLV
retention_offer_cost = 50  # $50 discount to prevent churn
retention_success_rate = 0.8  # 80% of offers succeed

# Calculate savings
high_risk_customers = (predictions_prob >= 0.75).sum()
retention_offers_cost = high_risk_customers * retention_offer_cost
retained_value = high_risk_customers * retention_success_rate * customer_lifetime_value
net_roi = retained_value - retention_offers_cost

print(f"\nScenario: Apply $50 retention offers to high-risk customers")
print(f"  High-risk customers (churn prob ≥ 75%): {high_risk_customers}")
print(f"  Total offer cost: ${retention_offers_cost:,.0f}")
print(f"  Expected retained value: ${retained_value:,.0f}")
print(f"  NET ROI: ${net_roi:,.0f}")
print(f"  ROI Percentage: {(net_roi / retention_offers_cost * 100):.1f}%")

print("\n✓ Neural network deployment successful!")
print("✓ Ready for production: predict churn for new customers")
print("✓ Continuously retrain monthly with new data for accuracy")

print("\n" + "="*80)