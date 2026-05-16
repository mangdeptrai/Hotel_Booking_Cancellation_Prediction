import pandas as pd

print("--- ĐANG LÀM SẠCH DỮ LIỆU ---")
# Đọc file dữ liệu thô ban đầu
df = pd.read_csv('hotel_reservations.csv')

# 1. Loại bỏ dòng trùng lặp
df_clean = df.drop_duplicates()

# 2. Xóa cột Booking_ID 
if 'Booking_ID' in df_clean.columns:
    df_clean = df_clean.drop('Booking_ID', axis=1)

# 3. Lọc khách ảo (0 người lớn, 0 trẻ em)
invalid_guests = (df_clean['no_of_adults'] == 0) & (df_clean['no_of_children'] == 0)
df_clean = df_clean[~invalid_guests]

# 4. Lọc giá phòng <= 0
df_clean = df_clean[df_clean['avg_price_per_room'] > 0]

print("Kích thước dữ liệu sau làm sạch:", df_clean.shape)

# XUẤT FILE TRUNG GIAN
df_clean.to_csv('hotel_clean_data.csv', index=False)
print("Đã lưu thành công file 'hotel_clean_data.csv'!")