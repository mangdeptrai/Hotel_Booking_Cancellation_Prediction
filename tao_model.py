import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Đang huấn luyện AI và tạo file hotel_model.pkl mới...")

# 1. Đọc dữ liệu sạch
df = pd.read_csv('hotel_clean_data.csv')

# 2. CHÚ Ý CHỖ NÀY: Thứ tự 6 cột phải khớp 100% với giao diện web
features = ['lead_time', 'avg_price_per_room', 'no_of_adults', 
            'no_of_special_requests', 'arrival_month', 'no_of_weekend_nights']

X = df[features]
y = df['booking_status']

# 3. Cho AI học
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# 4. Lưu đè bộ não mới
joblib.dump(model, 'hotel_model.pkl')

print("✅ Đã đẻ ra file 'hotel_model.pkl' thành công. Bạn hãy quay lại web bấm dự đoán nhé!")