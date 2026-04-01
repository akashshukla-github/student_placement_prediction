# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_csv("student_placement_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())


# ==============================
# 3. DATA CLEANING
# ==============================

# Drop unnecessary column
df.drop("Student_ID", axis=1, inplace=True)


# ==============================
# 4. ENCODING
# ==============================

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


# ==============================
# 5. EDA (GRAPHS)
# ==============================

# Placement Distribution
sns.countplot(x='Placement_Status', data=df)
plt.title("Placement Distribution")
plt.show()

# CGPA vs Placement
sns.boxplot(x='Placement_Status', y='CGPA', data=df)
plt.title("CGPA vs Placement")
plt.show()


# ==============================
# 6. SPLIT DATA
# ==============================

X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 7. LOGISTIC REGRESSION
# ==============================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))


# ==============================
# 8. RANDOM FOREST
# ==============================

rf_model = RandomForestClassifier(max_depth=5,random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))


# ==============================
# CONFUSION MATRIX
# ==============================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================

importances = rf_model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Feature Importance")
plt.show()