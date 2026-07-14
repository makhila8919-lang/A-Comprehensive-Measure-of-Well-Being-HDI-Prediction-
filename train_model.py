import pandas as pd
import joblib
from config import ENCODER_PATH

from config import MODEL_PATH
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("dataset/Human_Development_Data.csv")
print("\nDataset Columns:\n")

print(df.columns.tolist())

print(df.head())
print(df.info())

# -----------------------------
# Select Required Columns
# -----------------------------

df = df[
    [
        "Life expectancy at birth",
        "Expected years of schooling",
        "Mean years of schooling",
        "Gross national income (GNI) per capita",
        "HDI value"
    ]
]

# -----------------------------
# Handle Missing Values
# -----------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Convert HDI Score into Category
# -----------------------------

def categorize_hdi(hdi):

    if hdi >= 0.800:
        return "Very High"

    elif hdi >= 0.700:
        return "High"

    elif hdi >= 0.550:
        return "Medium"

    else:
        return "Low"


df["HDI Category"] = df["HDI value"].apply(categorize_hdi)

# Drop original target

df.drop("HDI value", axis=1, inplace=True)

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("HDI Category", axis=1)

y = df["HDI Category"]

# -----------------------------
# Encode Labels
# -----------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Save Encoder

joblib.dump(encoder, ENCODER_PATH)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# Train Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print("Accuracy:", accuracy)
print("=" * 50)

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel Saved Successfully")
