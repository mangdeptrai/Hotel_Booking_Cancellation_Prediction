import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

print("--- ĐANG ĐỌC DỮ LIỆU SẠCH ĐỂ VẼ BIỂU ĐỒ ---")
# ĐỌC FILE TRUNG GIAN VỪA ĐƯỢC TẠO RA TỪ FILE SỐ 1
df_clean = pd.read_csv('hotel_clean_data.csv')

# Biểu đồ 1: Tỷ lệ khách Hủy vs Không hủy
status_counts = df_clean['booking_status'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
        colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0, 0.1))
plt.title('Tỷ lệ hủy phòng và không hủy', fontsize=16, fontweight='bold')
plt.show() 

# Biểu đồ 2: Tháng nào có khách hủy phòng nhiều nhất?
plt.figure(figsize=(12, 6))
sns.countplot(data=df_clean, x='arrival_month', hue='booking_status', 
              palette='Set2', order=range(1, 13))
plt.title('Tình trạng hủy phòng theo từng tháng', fontsize=16, fontweight='bold')
plt.xlabel('Thang den')
plt.ylabel('So luong don dat')
plt.legend(title='Trang thai')
plt.show()

print("Hoàn thành quá trình vẽ biểu đồ!")