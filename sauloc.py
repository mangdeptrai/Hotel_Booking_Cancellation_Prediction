import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC DỮ LIỆU GỐC BAN ĐẦU
df = pd.read_csv('hotel_clean_data.csv')

# 2. LƯU LẠI CỘT BIẾN MỤC TIÊU NGUYÊN BẢN
target_original = df['booking_status'].copy()

# 3. MÃ HÓA BIẾN MỤC TIÊU (Canceled = 1, Not_Canceled = 0)
df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

# 4. MÃ HÓA BIẾN CHỮ (One-Hot Encoding bằng get_dummies)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# 5. CHUẨN HÓA DỮ LIỆU SỐ (Scaling) 
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
if 'booking_status' in numerical_cols:
    numerical_cols.remove('booking_status')

so_luong_cot_ban_dau = df_encoded.shape[1]

# Tiến hành Scale các cột số đầu vào
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 6. LỌC ĐẶC TRƯNG DỰA TRÊN MA TRẬN TƯƠNG QUAN (Feature Selection)
cor_matrix = df_encoded.corr()
cor_target = abs(cor_matrix["booking_status"])

# Lọc các đặc trưng có độ tương quan mạnh (Ngưỡng > 0.02)
relevant_features = cor_target[cor_target > 0.02].index.tolist()
if 'booking_status' not in relevant_features:
    relevant_features.append('booking_status')

# Tìm danh sách các cột bị loại bỏ để in ra màn hình báo cáo
all_features = df_encoded.columns.tolist()
removed_features = [col for col in all_features if col not in relevant_features]

# Tạo dataframe final tinh gọn (ĐÃ XÓA CÁC CỘT YẾU)
df_final = df_encoded[relevant_features].copy()

# Cột booking_status LUÔN LÀ số nguyên nhị phân (0 hoặc 1)
df_final['booking_status'] = target_original.map({'Canceled': 1, 'Not_Canceled': 0}).astype(int)

so_luong_cot_sau_loc = df_final.shape[1]

# 7. XUẤT FILE SẠCH CUỐI CÙNG CHO THÀNH VIÊN 4 VÀ 5
df_final.to_csv('hotel_reservations_final.csv', index=False)

print("   LỌC ĐẶC TRƯNG NHIỄU SẠCH SẼ (FEATURE SELECTION)")
print(f"Số lượng cột ban đầu (Sau khi One-Hot): {so_luong_cot_ban_dau} cột")
print(f"Số lượng cột giữ lại sau khi lọc (Ngưỡng > 0.02): {so_luong_cot_sau_loc} cột")

print(f"DANH SÁCH {len(removed_features)} CỘT BỊ LOẠI BỎ VÌ QUÁ YẾU (Tương quan < 0.02):")
for idx, col in enumerate(removed_features, 1):
    score = cor_matrix['booking_status'][col]
    print(f"  [{idx}] Loại bỏ: '{col}' (Độ tương quan: {score:.5f})")

print("Đã xuất file 'hotel_reservations_final.csv' THÀNH CÔNG")

# 8. VẼ BIỂU ĐỒ CỘT ĐỨNG (CHỈ GIỮ LẠI CÁC CỘT ĐÃ LỌC ĐỦ TIÊU CHUẨN)
final_cor_matrix = df_final.corr()
final_correlation = final_cor_matrix['booking_status'].drop('booking_status').sort_values(ascending=False)

plt.figure(figsize=(12, 6))
final_correlation.plot(kind='bar', color='seagreen', edgecolor='black') 
plt.title("Các đặc trưng tinh nhuệ ảnh hưởng tới việc Hủy phòng (Sau khi lọc > 0.02)", fontsize=14, fontweight='bold')
plt.ylabel("Hệ số tương quan (Correlation Coefficient)", fontsize=12)
plt.xlabel("Đặc trưng giữ lại (Selected Features)", fontsize=12)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()