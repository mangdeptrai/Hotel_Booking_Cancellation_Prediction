# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách vẽ biểu đồ cho đẹp mắt
sns.set_theme(style="whitegrid")

# 1. Đọc dữ liệu từ file CSV (Giả sử bạn 1 chưa làm xong SQL, bạn đọc thẳng file gốc)
df = pd.read_csv('hotel_reservations.csv')

# Xem nhanh 5 dòng đầu và thông tin tổng quan
print("Kích thước ban đầu:", df.shape)
print(df.head())
df.info()