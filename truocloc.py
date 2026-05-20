import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC DỮ LIỆU BAN ĐẦU
df = pd.read_csv('hotel_clean_data.csv')

# 3. MÃ HÓA BIẾN MỤC TIÊU (Canceled = 1, Not_Canceled = 0)
df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

# 4. PHÂN TÁCH BIẾN SỐ VÀ BIẾN CHỮ
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
if 'booking_status' in numerical_cols:
    numerical_cols.remove('booking_status')

# 5. MÃ HÓA BIẾN CHỮ (One-Hot Encoding bằng get_dummies)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# 6. CHUẨN HÓA DỮ LIỆU SỐ (Scaling bằng StandardScaler)
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
df_encoded['booking_status'] = df_encoded['booking_status'].astype(int)

# 7. XUẤT FILE TẠM THỜI ĐỂ LÀM BƯỚC TIẾP THEO
df_encoded.to_csv('hotel_encoded_temporary.csv', index=False)
print("Đã xuất file 'hotel_encoded_temporary.csv' thành công!")

# Tính toán tương quan thực tế (có âm có dương) với biến booking_status
cor_matrix = df_encoded.corr()
correlation = cor_matrix['booking_status'].drop('booking_status').sort_values(ascending=False)

print("5 ĐẶC TRƯNG CÓ TƯƠNG QUAN MẠNH NHẤT VỚI BIẾN MỤC TIÊU:")
# Lấy trị tuyệt đối để tìm độ mạnh thực sự, rồi lấy top 5 cột
top_5_features = abs(correlation).sort_values(ascending=False).head(5)
for idx, (col, val) in enumerate(top_5_features.items(), 1):
    original_val = correlation[col]
    print(f"  {idx}. Đặc trưng '{col}' với độ tương quan gốc: {original_val:.4f}")

# 8. VẼ BIỂU ĐỒ CỘT ĐỨNG 
plt.figure(figsize=(15, 7))
correlation.plot(kind='bar', color='royalblue', edgecolor='black')
plt.title("Mức độ ảnh hưởng của các đặc trưng tới việc Hủy phòng (Trước khi lọc)", fontsize=14, fontweight='bold')
plt.ylabel("Hệ số tương quan (Correlation Coefficient)", fontsize=12)
plt.xlabel("Đặc trưng (Features)", fontsize=12)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()