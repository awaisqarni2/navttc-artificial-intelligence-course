# ============================================================
# ELECTRICITY THEFT DETECTION - FINAL CLEAN VERSION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. DATA GENERATION + SAVE CSV
# ============================================================

# np.random.seed(42)
# n_samples = 2500

# df = pd.DataFrame()

# df['avg_consumption'] = np.random.normal(300, 100, n_samples).clip(50)
# df['peak_consumption'] = df['avg_consumption'] + np.random.normal(60, 30, n_samples)
# df['previous_month_consumption'] = df['avg_consumption'] + np.random.normal(0, 50, n_samples)

# df['weekly_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 20, n_samples)
# df['daily_variation'] = np.random.uniform(0, 1, n_samples)
# df['usage_spike'] = np.random.uniform(0, 1, n_samples)
# df['usage_drop_ratio'] = np.random.uniform(0, 1, n_samples)

# df['voltage'] = np.random.normal(220, 10, n_samples)
# df['current'] = np.random.normal(10, 3, n_samples)
# df['power_factor'] = np.random.uniform(0.7, 1.0, n_samples)
# df['frequency'] = np.random.normal(50, 0.5, n_samples)
# df['voltage_fluctuation'] = np.random.uniform(0, 1, n_samples)

# df['billing_amount'] = df['avg_consumption'] * np.random.uniform(0.1, 0.3, n_samples)
# df['avg_bill_last_3_months'] = df['billing_amount'] + np.random.normal(0, 50, n_samples)
# df['bill_difference'] = df['billing_amount'] - df['avg_bill_last_3_months']
# df['payment_delay_days'] = np.random.randint(0, 30, n_samples)
# df['unpaid_bills_count'] = np.random.randint(0, 5, n_samples)

# df['neighbor_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 40, n_samples)
# df['deviation_from_area'] = df['avg_consumption'] - df['neighbor_avg_consumption']
# df['line_loss'] = np.random.uniform(0, 0.3, n_samples)
# df['transformer_load'] = np.random.uniform(0.5, 1.5, n_samples)

# df['tamper_flag'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# # TARGET
# theft_score = (
#     0.25 * df['usage_drop_ratio'] +
#     0.2 * df['tamper_flag'] +
#     0.15 * abs(df['deviation_from_area']) +
#     0.15 * df['line_loss'] +
#     0.1 * (1 - df['power_factor']) +
#     0.15 * df['voltage_fluctuation']
# )

# df['theft'] = (theft_score > 0.6).astype(int)

# # SAVE CSV
# df.to_csv(r"electricity_theft_data.csv", index=False)

# 2. LOAD FROM CSV
# ============================================================
df = pd.read_csv(r"electricity_theft_data.csv")
# ===============================
# BASIC INFO
# ===============================
# print(df.shape)
# print(df.info())
# print(df.head())
# print(df.describe())
# exit()
# ===============================
# REMOVE DUPLICATES
# ===============================
# df = df.drop_duplicates()
# HANDLE MISSING VALUES
# ===============================
# Numeric columns → fill with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# exit()
# 3. FEATURE ENGINEERING
# ============================================================

df['power'] = df['voltage'] * df['current']
df['consumption_ratio'] = df['peak_consumption'] / df['avg_consumption']
df['consumption_change'] = (df['avg_consumption'] - df['previous_month_consumption']) / df['previous_month_consumption']

df['risk_score'] = df['usage_drop_ratio'] * 0.4 + df['tamper_flag'] * 0.6

df['bill_per_unit'] = df['billing_amount'] / (df['avg_consumption'] + 1)

# ============================================================
# 4. Z-SCORE CLEANING
# ============================================================

numeric_cols = df.drop('theft', axis=1).select_dtypes(include=[np.number]).columns
z_scores = np.abs(zscore(df[numeric_cols]))

df_clean = df[(z_scores < 3).all(axis=1)]

if df_clean['theft'].nunique() > 1:
    df = df_clean
else:
    print("Skipping Z-score cleaning")

# ============================================================
# 5. SIMPLE VISUAL (IMPORTANT)
# ============================================================

plt.figure()
plt.scatter(df['avg_consumption'], df['billing_amount'])
plt.title("Consumption vs Billing")
plt.xlabel("Consumption")
plt.ylabel("Billing")
plt.show()


# Correlation heatmap
plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()
# ============================================================
# 6. SPLIT
# ============================================================

X = df.drop('theft', axis=1)
y = df['theft']

# ============================================================
# 7. SMOTE
# ============================================================

# BEFORE
plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.show()

smote = SMOTE()
X, y = smote.fit_resample(X, y)

# AFTER
plt.figure()
pd.Series(y).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.show()

# ============================================================
# 8. SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================================================
# 10. MODEL
# ============================================================

model = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# 11. EVALUATION
# ============================================================

y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("Precision:", precision_score(y_test, y_pred)*100)
print("Recall:", recall_score(y_test, y_pred)*100)
print("F1:", f1_score(y_test, y_pred)*100)

# CONFUSION MATRIX (VISUAL)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

plt.figure()
plt.barh(df.drop('theft', axis=1).columns, model.feature_importances_)
plt.title("Feature Importance")
plt.show()

# ============================================================
# 13. PREDICTION
# ============================================================

predictTheft = X_test[0].reshape(1, -1)
print("Prediction:", model.predict(predictTheft)[0])

# ============================================================
# 14. USER INPUT PREDICTION
# ============================================================

print("\nEnter Customer Data to Predict Theft:\n")
try:
    avg_consumption = float(input("Avg Consumption: "))
    peak_consumption = float(input("Peak Consumption: "))
    previous_month_consumption = float(input("Previous Month Consumption: "))

    weekly_avg_consumption = float(input("Weekly Avg Consumption: "))
    daily_variation = float(input("Daily Variation (0-1): "))
    usage_spike = float(input("Usage Spike (0-1): "))
    usage_drop_ratio = float(input("Usage Drop Ratio (0-1): "))

    voltage = float(input("Voltage: "))
    current = float(input("Current: "))
    power_factor = float(input("Power Factor (0-1): "))
    frequency = float(input("Frequency: "))
    voltage_fluctuation = float(input("Voltage Fluctuation (0-1): "))

    billing_amount = float(input("Billing Amount: "))
    avg_bill_last_3_months = float(input("Avg Bill Last 3 Months: "))
    bill_difference = billing_amount - avg_bill_last_3_months

    payment_delay_days = int(input("Payment Delay Days: "))
    unpaid_bills_count = int(input("Unpaid Bills Count: "))

    neighbor_avg_consumption = float(input("Neighbor Avg Consumption: "))
    deviation_from_area = avg_consumption - neighbor_avg_consumption

    line_loss = float(input("Line Loss (0-1): "))
    transformer_load = float(input("Transformer Load: "))

    tamper_flag = int(input("Tamper Flag (0 or 1): "))

    # FEATURE ENGINEERING (SAME AS TRAINING)
    # ===============================

    power = voltage * current
    consumption_ratio = peak_consumption / avg_consumption
    consumption_change = (avg_consumption - previous_month_consumption) / previous_month_consumption
    risk_score = usage_drop_ratio * 0.4 + tamper_flag * 0.6
    bill_per_unit = billing_amount / (avg_consumption + 1)

    # CREATE INPUT ARRAY
    # ===============================

    input_data = pd.DataFrame([{
    'avg_consumption': avg_consumption,
    'peak_consumption': peak_consumption,
    'previous_month_consumption': previous_month_consumption,
    'weekly_avg_consumption': weekly_avg_consumption,
    'daily_variation': daily_variation,
    'usage_spike': usage_spike,
    'usage_drop_ratio': usage_drop_ratio,
    'voltage': voltage,
    'current': current,
    'power_factor': power_factor,
    'frequency': frequency,
    'voltage_fluctuation': voltage_fluctuation,
    'billing_amount': billing_amount,
    'avg_bill_last_3_months': avg_bill_last_3_months,
    'bill_difference': bill_difference,
    'payment_delay_days': payment_delay_days,
    'unpaid_bills_count': unpaid_bills_count,
    'neighbor_avg_consumption': neighbor_avg_consumption,
    'deviation_from_area': deviation_from_area,
    'line_loss': line_loss,
    'transformer_load': transformer_load,
    'tamper_flag': tamper_flag,
    'power': power,
    'consumption_ratio': consumption_ratio,
    'consumption_change': consumption_change,
    'risk_score': risk_score,
    'bill_per_unit': bill_per_unit
}])
    # SCALE INPUT
    # ===============================

    input_scaled = scaler.transform(input_data)
    # PREDICT
    # ===============================
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        print("\n⚠️ Theft Detected!")
    else:
        print("\n✅ No Theft Detected")
 # ===============================
    # SAVE TO CSV
    # ===============================
    import os
    file_path = "theft_predictions_data.csv"

    # If file exists → append, else create with header
    if os.path.exists(file_path):
        input_data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        input_data.to_csv(file_path, mode='w', header=True, index=False)

    print(f"\n📁 Data saved to {file_path}")
except Exception as e:
    print("An error occurred:", e)