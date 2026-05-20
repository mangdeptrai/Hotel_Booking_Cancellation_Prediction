import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Đọc dữ liệu và xử lý nhanh
df = pd.read_csv('hotel_clean_data.csv')
df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

# One-Hot Encoding
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe_encoded = ohe.fit_transform(df[categorical_cols])
df_ohe = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_cols))

# Scaling
numerical_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                  'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month', 
                  'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 
                  'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
scaler = StandardScaler()
df_scaled_num = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Gộp dữ liệu hoàn chỉnh
df_processed = pd.concat([df_scaled_num, df_ohe, df['booking_status']], axis=1)

# --- BIỂU ĐỒ 1: MA TRẬN TƯƠNG QUAN (ĐÃ ẨN SỐ BÊN TRONG Ô) ---
plt.figure(figsize=(16, 12))
# Lấy top các biến có tương quan cao nhất với booking_status để vẽ cho đỡ rối
top_corr_features = df_processed.corr()['booking_status'].sort_values(ascending=False).index

# MẸO Ở ĐÂY: Đổi annot=True thành annot=False để biến mất toàn bộ chữ/số trong ô vuông bà nhé
sns.heatmap(df_processed[top_corr_features].corr(), annot=False, cmap="coolwarm", linewidths=0.5)

plt.title("Ma Trận Tương Quan Giữa Các Đặc Trưng", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300) # Lưu biểu đồ thành file ảnh sạch sẽ
plt.show()

# --- BIỂU ĐỒ 2: ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (KEEP UNCHANGED) ---
X = df_processed.drop(columns=['booking_status'])
y = df_processed['booking_status']
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({'Feature': X.columns, 'Scores': selector.scores_}).sort_values(by='Scores', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(feature_scores['Feature'], feature_scores['Scores'], color='skyblue', edgecolor='black')
plt.xlabel('Điểm số ANOVA F-Value (Càng cao càng quan trọng)', fontsize=12)
plt.ylabel('Các đặc trưng (Features)', fontsize=12)
plt.title('Bảng Xếp Hạng Độ Quan Trọng Của Đặc Trưng (SelectKBest)', fontsize=14, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()