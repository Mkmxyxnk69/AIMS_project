import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
data = pd.read_csv("/Users/mayankkumar/Desktop/python 12.46.49 PM/drone_gestures_data.csv", header=None)

# 2. Features (Coordinates) and Labels (Gesture Name)
X = data.iloc[:, :-1] # Saare coordinates
y = data.iloc[:, -1]  # Last column (Gesture Name)

# 3. Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train KNN Model
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)

# 5. Check Accuracy
predictions = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Save the Model
joblib.dump(model, "gesture_model_3.pkl")
print("🚀 Model saved as gesture_model.pkl")