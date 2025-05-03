import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import sklearn

# Check scikit-learn version to handle the OneHotEncoder argument accordingly
scikit_version = sklearn.__version__

# Load dataset
df = pd.read_csv("input dataset.csv")

# Define features and target
X = df.drop("target", axis=1)
y = df["target"]

# Categorical columns
categorical_columns = ['cp', 'restecg', 'slope', 'thal', 'ca']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Determine the correct argument for OneHotEncoder based on scikit-learn version
if scikit_version >= '1.2':
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # For scikit-learn >= 1.2
else:
    encoder = OneHotEncoder(drop='first', sparse=False)  # For scikit-learn < 1.2

# One-hot encode
X_encoded = encoder.fit_transform(X[categorical_columns])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Final training data
X_final = pd.concat([X[numerical_columns].reset_index(drop=True), X_encoded_df], axis=1)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_final, y)

# Save model, encoder, and feature names
joblib.dump(model, "xgb_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(X_final.columns.tolist(), "feature_columns.pkl")

print("✅ Model training complete. Files saved!")
