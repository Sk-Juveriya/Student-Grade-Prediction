# =========================================================
# STUDENT GRADE PREDICTION (RANDOM DATASET)
# ========================================================= 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# STEP 1: GENERATE RANDOM DATASET
# =========================================================

np.random.seed(42)

df = pd.DataFrame({
    "Study_Hours": np.random.randint(1, 11, 120),
    "Attendance": np.random.randint(40, 101, 120),
    "Assignments": np.random.randint(1, 11, 120)
})

# Create realistic grade pattern
df["Final_Grade"] = (
    df["Study_Hours"] * 5 +
    df["Attendance"] * 0.4 +
    df["Assignments"] * 3 +
    np.random.randint(-5, 6, 120)
)

df["Final_Grade"] = np.clip(df["Final_Grade"], 0, 100)

print("📊 Dataset Preview:")
print(df.head())


# =========================================================
# STEP 2: FEATURES & TARGET
# =========================================================

X = df[["Study_Hours", "Attendance", "Assignments"]]
y = df["Final_Grade"]


# =========================================================
# STEP 3: TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# STEP 4: MODEL
# =========================================================

model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)


# =========================================================
# STEP 5: PREDICTION
# =========================================================

y_pred = model.predict(X_test)

print("\n📈 Predicted:")
print(y_pred[:10])

print("\n✅ Actual:")
print(y_test.values[:10])


# =========================================================
# STEP 6: EVALUATION
# =========================================================

rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n🎯 RMSE:", round(rmse, 2))
print("🎯 R2 Score:", round(r2, 2))
print("🎯 Accuracy:", round(r2 * 100, 2), "%")


# =========================================================
# STEP 7: VISUALIZATION
# =========================================================

plt.figure(figsize=(15,4))

#Plot 1: STUDY HOURS VS GRADE
plt.subplot(1,3,1)
plt.scatter(df["Study_Hours"], df["Final_Grade"])
plt.title("Study Hours vs Grade")
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.grid()

#PLOT 2: ATTENDANCE VS GRADE
plt.subplot(1,3,2)
plt.scatter(df["Attendance"], df["Final_Grade"], color="green")
plt.title("Attendance vs Grade")
plt.xlabel("Attendance")
plt.ylabel("Final Grade")
plt.grid()

#PLOT 3: ASSIGNMENTS VS GRADE
plt.subplot(1,3,3)
plt.scatter(df["Assignments"], df["Final_Grade"], color="red")
plt.title("Assignments vs Grade")
plt.xlabel("Assignments")
plt.ylabel("Final Grade")
plt.grid()

plt.tight_layout()
plt.show()


# =========================================================
# STEP 8: NEW STUDENT PREDICTION
# =========================================================

new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [85],
    "Assignments": [7]
})

pred = model.predict(new_student)

print("\n🧑‍🎓 Predicted Grade for New Student:")
print(round(pred[0], 2))


# =========================================================
# STEP 9: CONCLUSION
# =========================================================

print("\n📚 CONCLUSION")
print("""
This project demonstrates a complete Machine Learning workflow:

- Random dataset generation
- Feature selection
- Decision Tree Regression model
- Prediction and evaluation
- Data visualization

It shows how student performance can be predicted using ML.
""")
