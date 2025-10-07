# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# STEP 1: Read the Data
try:
    df = pd.read_csv('bmi.csv')
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: 'bmi.csv' file not found. Please check the file path.")
    exit()

# STEP 2: Clean the Data Set
original_rows = df.shape[0]
df_clean = df.dropna()
dropped_rows = original_rows - df_clean.shape[0]
print(f"Removed {dropped_rows} rows with missing values ({dropped_rows/original_rows:.1%} of data)")

# STEP 3: Encode Categorical Features
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    print(f"Encoded categorical column: {col} → {len(le.classes_)} unique values")

# STEP 4: Feature Scaling
if 'Index' in df_clean.columns:
    numeric_cols = [col for col in df_clean.columns if col != 'Index']
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    print(f"Scaled {len(numeric_cols)} numeric features to range [0,1]")
else:
    print("Warning: 'Index' column not found. Please check your data.")
    exit()

# STEP 5: Feature Selection
X = df_clean.drop('Index', axis=1)
y = df_clean['Index']
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support(indices=True)]
print(f"Selected top features: {', '.join(selected_columns)}")

# Combine selected features with the target column
df_selected = pd.concat([df_clean[selected_columns], y], axis=1)

# STEP 6: Save the resulting data
output_file = 'bmi_selected_scaled.csv'
df_selected.to_csv(output_file, index=False)
print(f"Saved processed data to '{output_file}' ({df_selected.shape[0]} rows, {df_selected.shape[1]} columns)")

```

# RESULT/OUTPUT:

<img width="745" height="147" alt="Screenshot 2025-10-07 214504" src="https://github.com/user-attachments/assets/2d903c70-e533-471f-9afb-de278900b0bb" />

