import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("--- CHẠY MODEL CÂY QUYẾT ĐỊNH (BASELINE) ---")

# 1. Đọc dữ liệu đã được làm sạch của chính bạn
df = pd.read_csv('hotel_clean_data.csv')

# 2. Tách biến mục tiêu (y) và các đặc trưng (X)
y = df['booking_status']
X = df.drop('booking_status', axis=1)

# Mã hóa nhanh các cột chữ thành số (One-Hot Encoding cơ bản) để cây quyết định đọc được
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Chia tập dữ liệu: 80% để học, 20% để kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Huấn luyện Model với độ sâu giới hạn là 5
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Dự đoán và chấm điểm
y_pred = model.predict(X_test)
diem_so = accuracy_score(y_test, y_pred)

print(f"\n=> ĐỘ CHÍNH XÁC CỦA MODEL CƠ BẢN LÀ: {diem_so * 100:.2f}%")