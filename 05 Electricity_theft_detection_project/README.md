============================================================
ELECTRICITY THEFT DETECTION SYSTEM
FINAL CLEAN VERSION
============================================================

Author:
Machine Learning Project

============================================================
PROJECT DESCRIPTION
============================================================

This project is an Electricity Theft Detection System developed
using Machine Learning with Python.

The system analyzes electricity consumption patterns and predicts
whether a customer may be involved in electricity theft.

The project uses:
- Data preprocessing
- Feature engineering
- Outlier removal
- Data balancing using SMOTE
- Random Forest Classification
- Visualization and evaluation metrics

The model can also take manual customer input and predict
possible theft in real-time.

============================================================
FEATURES
============================================================

✔ Synthetic dataset generation
✔ CSV dataset loading
✔ Missing value handling
✔ Feature engineering
✔ Outlier removal using Z-score
✔ Data visualization
✔ SMOTE balancing
✔ Feature scaling
✔ Random Forest training
✔ Model evaluation
✔ Confusion matrix visualization
✔ Feature importance graph
✔ Manual customer prediction
✔ Save prediction data into CSV

============================================================
TECHNOLOGIES USED
============================================================

Programming Language:
- Python

Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- imbalanced-learn

============================================================
INSTALLATION
============================================================

1. Install Python (3.9 or above recommended)

2. Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn

============================================================
DATASET
============================================================

The project uses a CSV dataset:

electricity_theft_data_p3.csv

The dataset contains features such as:
- Average consumption
- Peak consumption
- Voltage
- Current
- Billing amount
- Line loss
- Tamper flag
- Power factor
- Transformer load
- etc.

Target Column:
- theft
  0 = No Theft
  1 = Theft

============================================================
HOW TO RUN
============================================================

1. Open terminal or command prompt

2. Navigate to project folder

3. Run the script:

python filename.py

============================================================
WORKFLOW
============================================================

1. Load dataset from CSV
2. Handle missing values
3. Create engineered features
4. Remove outliers using Z-score
5. Visualize data
6. Apply SMOTE for balancing
7. Scale features
8. Split train/test data
9. Train Random Forest model
10. Evaluate performance
11. Predict theft
12. Accept user input for live prediction
13. Save prediction data into CSV

============================================================
FEATURE ENGINEERING
============================================================

New features created:
- power
- consumption_ratio
- consumption_change
- risk_score
- bill_per_unit

These features improve model performance.

============================================================
VISUALIZATIONS
============================================================

The system generates:
- Consumption vs Billing scatter plot
- Correlation heatmap
- SMOTE class distribution charts
- Confusion matrix
- Feature importance graph

============================================================
MODEL DETAILS
============================================================

Algorithm:
- RandomForestClassifier

Parameters:
- n_estimators = 400
- max_depth = 12
- random_state = 42

============================================================
EVALUATION METRICS
============================================================

The project evaluates:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

============================================================
USER INPUT PREDICTION
============================================================

The program allows users to manually enter customer data.

After input:
- Data is processed
- Features are engineered
- Data is scaled
- Prediction is generated

Output:
⚠️ Theft Detected!
or
✅ No Theft Detected

============================================================
OUTPUT FILE
============================================================

Predicted customer data is saved in:

theft_predictions_data.csv

============================================================
IMPORTANT NOTES
============================================================

1. Ensure dataset path is correct in the code.

Example:
df = pd.read_csv("path_to_dataset.csv")

2. If dataset generation code is commented,
   existing CSV dataset must be available.

3. SMOTE requires at least two classes in target data.

============================================================
FUTURE IMPROVEMENTS
============================================================

Possible future enhancements:
- Deep Learning models
- Real smart meter integration
- Web dashboard
- Real-time monitoring
- API integration
- Fraud alert system

============================================================
LICENSE
============================================================

This project is for educational and research purposes.

============================================================
END OF README
============================================================