import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# TRƯỜNG HỢP 1: DỮ LIỆU ĐÃ QUA XỬ LÝ (Mã hóa + Scale + Lọc đặc trưng) 
df_final = pd.read_csv('hotel_reservations_final.csv')

X_f = df_final.drop(columns=['booking_status'])
y_f = df_final['booking_status']

# Chia tập dữ liệu Train/Test theo tỷ lệ 80/20 với random_state=42 để đối chứng khách quan
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression trên tập dữ liệu tinh nhuệ
model_f = LogisticRegression(max_iter=1000)
model_f.fit(X_train_f, y_train_f)

# Tính toán độ chính xác (Accuracy) sau khi đã xử lý dữ liệu số & lọc nhiễu
acc_final = accuracy_score(y_test_f, model_f.predict(X_test_f))

# TRƯỜNG HỢP 2: DỮ LIỆU GỐC CHƯA LỌC & CHƯA SCALE KỸ 
# Đọc trực tiếp từ file dữ liệu thô sạch ban đầu
df_raw = pd.read_csv('hotel_clean_data.csv')

# Chuyển đổi biến mục tiêu thành dạng nhị phân 0 và 1 để mô hình phân loại hoạt động được
df_raw['booking_status'] = df_raw['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

# Chuyển đổi các biến phân loại dạng chữ thành số bằng One-Hot Encoding (Giữ nguyên toàn bộ cột nhiễu, KHÔNG SCALE)
df_raw = pd.get_dummies(df_raw, drop_first=True, dtype=int)

X_r = df_raw.drop(columns=['booking_status'])
y_r = df_raw['booking_status']

# Chia tập dữ liệu Train/Test tương tự Trường hợp 1
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression trên tập dữ liệu thô (chưa scale, chưa lọc cột yếu)
model_r = LogisticRegression(max_iter=1000)
model_r.fit(X_train_r, y_train_r)

# Tính toán độ chính xác (Accuracy) của mô hình thô đối chứng
acc_raw = accuracy_score(y_test_r, model_r.predict(X_test_r))

print("        KẾT QUẢ ĐỐI CHỨNG HIỆU QUẢ CỦA BƯỚC FEATURE ENGINEERING        ")
print(f"Kích thước không gian đặc trưng (Số lượng cột đầu vào X):")
print(f"Dữ liệu thô chưa lọc (hotel_clean_data) : {X_r.shape[1]} cột")
print(f"Dữ liệu tinh lọc (hotel_reservations_final) : {X_f.shape[1]} cột")
print(f"Kết quả: Bạn đã tinh gọn và loại bỏ được {X_r.shape[1] - X_f.shape[1]} cột nhiễu!")

print(f"Độ chính xác tổng thể của mô hình (Accuracy Score):")
print(f"Mô hình chạy trên Dữ liệu thô ban đầu    : {acc_raw:.4f} ({acc_raw:.2%})")
print(f"Mô hình chạy trên Dữ liệu đã xử lý kỹ   : {acc_final:.4f} ({acc_final:.2%})")

# Tính toán độ chênh lệch cải thiện thực tế
cai_thien = acc_final - acc_raw
if cai_thien > 0:
    print(f"Hiệu suất mô hình tăng trưởng tích cực: +{cai_thien:.4f} (+{cai_thien:.2%})")
else:
    print(f"Hiệu suất mô hình thay đổi không đáng kể: {cai_thien:.4f} ({cai_thien:.2%})")

# VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN 
categories = ['Dữ liệu gốc thô\n(Chưa lọc & Chưa Scale)', 'Dữ liệu tinh nhuệ\n(Đã Feature Engineering)']
scores = [acc_raw, acc_final]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, scores, color=['#E67E22', '#2C3E50'], edgecolor='black', width=0.5)

plt.title('So sánh độ chính xác (Accuracy) của Logistic Regression', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Độ chính xác (Accuracy Score)', fontsize=11)
plt.ylim(0, 1.1)  
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2%}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()