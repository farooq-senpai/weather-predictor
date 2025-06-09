import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your CSV
df = pd.read_csv("seattle-weather.csv")  # replace with the actual filename

# Encode labels
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

# Features and target
X = df[["precipitation", "temp_max", "temp_min", "wind"]]
y = df["weather"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "weather_model.pkl")
joblib.dump(le, "weather_label_encoder.pkl")
