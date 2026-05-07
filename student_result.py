#student result predition
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5],
    "Marks": [35, 40, 50, 60, 75]
}

df = pd.DataFrame(data)

# Features and target
X = df[["StudyHours"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
hours = pd.DataFrame([[6]], columns=["StudyHours"])
predicted_marks = model.predict(hours)
print("Predicted Marks:", predicted_marks[0])
