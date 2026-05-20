# =========================================================
# SO SANH CAC MO HINH - THANH VIEN 4 vs THANH VIEN 5
# =========================================================

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Mô hình cơ bản (TV4)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Mô hình nâng cao (TV5)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ======================
# 1. DOC DU LIEU
# ======================

df = pd.read_csv(
    "Hotel_Booking_Cancellation_Prediction/hotel_reservations_final.csv"
)

# ======================
# 2. TACH X VA y
# ======================

X = df.drop(columns=["booking_status"])
y = df["booking_status"]

# ======================
# 3. CHIA TRAIN / TEST
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 4. CAC MO HINH CO BAN - THANH VIEN 4
# =========================================================

print("\n================================================")
print("HUAN LUYEN CAC MO HINH CO BAN (TV4)")
print("================================================")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# =========================================================
# 5. RANDOM FOREST - THANH VIEN 5
# =========================================================

print("\n================================================")
print("RANDOM FOREST TOI UU (TV5)")
print("================================================")

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# =========================================================
# 6. TINH RECALL CHO CAC MO HINH
# =========================================================

# Logistic Regression
y_pred_lr = lr_model.predict(X_test)

recall_lr = classification_report(
    y_test,
    y_pred_lr,
    output_dict=True
)["1"]["recall"] * 100

# Decision Tree
y_pred_dt = dt_model.predict(X_test)

recall_dt = classification_report(
    y_test,
    y_pred_dt,
    output_dict=True
)["1"]["recall"] * 100

# KNN
y_pred_knn = knn_model.predict(X_test)

recall_knn = classification_report(
    y_test,
    y_pred_knn,
    output_dict=True
)["1"]["recall"] * 100

# Random Forest threshold mặc định = 0.5
y_pred_rf_default = rf_model.predict(X_test)

recall_rf_default = classification_report(
    y_test,
    y_pred_rf_default,
    output_dict=True
)["1"]["recall"] * 100

# Random Forest threshold cải tiến = 0.3
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

y_pred_rf_custom = (y_pred_prob_rf >= 0.3).astype(int)

recall_rf_custom = classification_report(
    y_test,
    y_pred_rf_custom,
    output_dict=True
)["1"]["recall"] * 100

# =========================================================
# 7. TAO BANG SO SANH
# =========================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest (Threshold 0.5)",
        "Random Forest (Threshold 0.3)"
    ],
    "Recall (%)": [
        recall_lr,
        recall_dt,
        recall_knn,
        recall_rf_default,
        recall_rf_custom
    ]
})

print("\n================================================")
print("BANG SO SANH RECALL")
print("================================================")

print(comparison_df)

# =========================================================
# 8. VE BIEU DO SO SANH
# =========================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

colors = [
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#ff7f0e"
]

bars = plt.bar(
    comparison_df["Model"],
    comparison_df["Recall (%)"],
    color=colors
)

# Hien thi gia tri tren moi cot
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title(
    "SO SANH CHI SO RECALL GIUA CAC MO HINH",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("Recall (%)")

plt.xticks(rotation=8)

plt.ylim(0, 110)

plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.show()

# =========================================================
# 9. NHAN XET
# =========================================================

print("\n================================================")
print("NHAN XET")
print("================================================")

print("- Random Forest cua TV5 cho Recall cao hon cac mo hinh co ban.")
print("- Dieu chinh threshold tu 0.5 xuong 0.3 giup tang kha nang phat hien khach huy phong.")
print("- Mo hinh Random Forest sau toi uu phu hop hon voi bai toan du doan huy dat phong.")