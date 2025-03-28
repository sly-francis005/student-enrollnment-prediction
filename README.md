README.md# student-enrollnment-predictionimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Creating a mock dataset with random values
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "GPA": np.round(np.random.uniform(2.0, 4.0, n_samples), 2),
    "test_scores": np.random.randint(50, 100, n_samples),
    "attendance": np.random.randint(60, 100, n_samples),
    "parental_education": np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n_samples),
    "scholarship": np.random.choice([0, 1], n_samples),
    "previous_enrollments": np.random.randint(0, 3, n_samples),
    "enrolled": np.random.choice([0, 1], n_samples)
})

# Encoding categorical features
data = pd.get_dummies(data, columns=["parental_education"], drop_first=True)

# Feature selection
features = [col for col in data.columns if col != "enrolled"]
target = "enrolled"

X = data[features]
y = data[target]

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
print(feature_importance.sort_values(by="Importance", ascending=False))
