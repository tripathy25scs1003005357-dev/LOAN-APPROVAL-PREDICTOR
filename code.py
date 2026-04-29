# Loan Approval Prediction using CSV dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("loan_approval_1000_dataset.csv")

print("Dataset Loaded Successfully ✅")
print(df.head())

# -------------------------------
# 2. Features & Target
# -------------------------------
X = df[["Income", "Credit_Score", "Loan_Amount", "Employment"]]
y = df["Approved"]

# -------------------------------
# 3. Feature Scaling (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Model Training
# -------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# -------------------------------
# 6. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. User Input Prediction
# -------------------------------
print("\n--- Enter New Data for Prediction ---")

income = float(input("Enter Income: "))
credit_score = float(input("Enter Credit Score: "))
loan_amount = float(input("Enter Loan Amount: "))
employment = int(input("Employment? (1 = Yes, 0 = No): "))

# Convert input to DataFrame (FIXED)
new_data = pd.DataFrame({
    "Income": [income],
    "Credit_Score": [credit_score],
    "Loan_Amount": [loan_amount],
    "Employment": [employment]
})

# Apply SAME scaling (VERY IMPORTANT)
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

# -------------------------------
# 8. Output Result
# -------------------------------
if prediction[0] == 1:
    print("\n✅ Loan Approved")
else:
    print("\n❌ Loan Rejected")