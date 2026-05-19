# =========================================================
# HOTEL BOOKING CANCELLATION PREDICTION
# THÀNH VIÊN 4 - MACHINE LEARNING
# =========================================================

# ======================
# 1. IMPORT THƯ VIỆN
# ======================

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ======================
# 2. ĐỌC FILE DỮ LIỆU
# ======================

df = pd.read_csv("hotel_reservations_final.csv")

# ======================
# 3. TÁCH X VÀ y
# ======================

# X: dữ liệu đầu vào
X = df.drop("booking_status", axis=1)

# y: nhãn cần dự đoán
y = df["booking_status"]

# ======================
# 4. CHIA TRAIN / TEST
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 5. LOGISTIC REGRESSION
# =========================================================

print("\n================================================")
print("LOGISTIC REGRESSION")
print("================================================")

lr_model = LogisticRegression(max_iter=1000)

# Huấn luyện mô hình
lr_model.fit(X_train, y_train)

# Dự đoán
y_pred_lr = lr_model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred_lr))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# =========================================================
# 6. DECISION TREE
# =========================================================

print("\n================================================")
print("DECISION TREE")
print("================================================")

dt_model = DecisionTreeClassifier(random_state=42)

# Huấn luyện
dt_model.fit(X_train, y_train)

# Dự đoán
y_pred_dt = dt_model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred_dt))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# =========================================================
# 7. KNN
# =========================================================

print("\n================================================")
print("K-NEAREST NEIGHBORS (KNN)")
print("================================================")

knn_model = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện
knn_model.fit(X_train, y_train)

# Dự đoán
y_pred_knn = knn_model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred_knn))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# =========================================================
# 8. SO SÁNH CÁC MÔ HÌNH
# =========================================================

print("\n================================================")
print("SO SÁNH CÁC MÔ HÌNH")
print("================================================")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_knn)
    ]
})

print(comparison)

# =========================================================
# 9. MODEL TỐT NHẤT
# =========================================================

best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\n================================================")
print("MÔ HÌNH TỐT NHẤT")
print("================================================")

print(f"Model tốt nhất: {best_model['Model']}")
print(f"Accuracy cao nhất: {best_model['Accuracy']:.4f}")