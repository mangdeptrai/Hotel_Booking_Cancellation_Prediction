# ======================
# 1. IMPORT THƯ VIỆN
# ======================

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ======================
# 2. ĐỌC FILE DỮ LIỆU
# ======================

# Đọc file dữ liệu sạch đã được xử lý từ các bạn trước
df = pd.read_csv("Hotel_Booking_Cancellation_Prediction/hotel_reservations_final.csv")

# ======================
# 3. TÁCH X VÀ y
# ======================

# X: các đặc trưng đầu vào (loại bỏ cột nhãn)
X = df.drop("booking_status", axis=1)

# y: nhãn cần dự đoán (booking_status)
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
# 5. RANDOM FOREST + GRID SEARCH
# =========================================================

print("\n================================================")
print("RANDOM FOREST + GRID SEARCH OPTIMIZATION")
print("================================================")

# Khởi tạo mô hình Random Forest
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

# Lưới tham số để tìm bộ tối ưu
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# GridSearchCV tối ưu theo Recall
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="recall",
    n_jobs=2
)

# Huấn luyện mô hình
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

print("\nTham so toi uu nhat tim duoc:")
print(grid_search.best_params_)

# =========================================================
# 6. KẾT QUẢ VỚI THRESHOLD MẶC ĐỊNH (0.5)
# =========================================================

print("\n================================================")
print("KET QUA VOI THRESHOLD MAC DINH (0.5)")
print("================================================")

y_pred_default = best_model.predict(X_test)

print(classification_report(y_test, y_pred_default))

# =========================================================
# 7. CẢI TIẾN: HẠ THRESHOLD XUỐNG 0.3
# =========================================================

print("\n================================================")
print("KET QUA VOI THRESHOLD CAI TIEN (0.3)")
print("================================================")

# Lấy xác suất dự đoán lớp 1
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Áp dụng threshold mới
y_pred_custom = (y_pred_prob >= 0.3).astype(int)

print(classification_report(y_test, y_pred_custom))

# =========================================================
# 8. ĐÁNH GIÁ CHỈ SỐ CHÍNH
# =========================================================

print("\n================================================")
print("DANH GIA CHI SO CHINH")
print("================================================")

# Accuracy và ROC AUC
print(f"Accuracy  : {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")

# In Recall riêng cho lớp hủy phòng
report = classification_report(
    y_test,
    y_pred_custom,
    output_dict=True
)

print(f"Recall cua lop Huy phong: {report['1']['recall']:.4f}")

# =========================================================
# 9. VẼ CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred_custom)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Canceled", "Canceled"],
    yticklabels=["Not Canceled", "Canceled"]
)

plt.title("Confusion Matrix (Threshold = 0.3)")
plt.ylabel("Thực tế (Actual)")
plt.xlabel("Dự đoán (Predicted)")

plt.show()

# =========================================================
# 10. VẼ ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

roc_auc_val = auc(fpr, tpr)

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc_val:.2f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("Biểu đồ ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# =========================================================
# 11. NHẬN XÉT
# =========================================================

print("\n================================================")
print("NHAN XET")
print("================================================")

print("- Random Forest sau toi uu cho ket qua tot.")
print("- Giam threshold tu 0.5 xuong 0.3 giup tang Recall.")
print("- Mo hinh phat hien duoc nhieu khach co nguy co huy phong hon.")
print("- ROC AUC cao cho thay mo hinh co kha nang phan biet tot.")
