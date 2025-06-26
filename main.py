import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset (for regression; for classification use another dataset)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print("X:", X)
print("y:", y)

# For classification: binarize the target (e.g., above/below median) 
# Converts the regression target into a binary classification problem
# -> Targets above the median are labeled 1, others 0.
y_binary = (y > np.median(y)).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Optional: Standard scaling -> Standardizes features to have mean 0 and variance 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
