import streamlit as st
import pandas as pd
import joblib

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Dự đoán Hủy phòng", page_icon="🏨", layout="centered")

st.title("🏨 Hệ thống Dự đoán Hủy phòng Khách sạn")
st.write("---")

# 2. TẠO KHUNG NHẬP DỮ LIỆU TỪ NGƯỜI DÙNG
st.subheader("Nhập thông tin khách hàng:")

col1, col2 = st.columns(2)
with col1:
    lead_time = st.number_input("Thời gian đặt trước (ngày):", min_value=0, value=30)
    avg_price_per_room = st.number_input("Giá phòng trung bình ($):", min_value=0.0, value=100.0)
    no_of_adults = st.number_input("Số người lớn:", min_value=1, value=2)

with col2:
    no_of_special_requests = st.number_input("Số yêu cầu đặc biệt:", min_value=0, value=1)
    arrival_month = st.selectbox("Tháng đến (1-12):", list(range(1, 13)))
    no_of_weekend_nights = st.number_input("Số đêm cuối tuần:", min_value=0, value=1)

# 3. NÚT BẤM DỰ ĐOÁN
if st.button("🔍 Phân tích và Dự đoán"):
    st.write("---")
    
    try:
        # Gọi "bộ não" ra để làm việc
        model = joblib.load('hotel_model.pkl')
        
        # Đóng gói 6 câu trả lời của người dùng thành một bảng dữ liệu chuẩn
        input_data = pd.DataFrame([[lead_time, avg_price_per_room, no_of_adults, no_of_special_requests, arrival_month, no_of_weekend_nights]], 
                                columns=['lead_time', 'avg_price_per_room', 'no_of_adults', 'no_of_special_requests', 'arrival_month', 'no_of_weekend_nights'])
        
        # Cho AI dự đoán
        ket_qua = model.predict(input_data)
        
        # In kết quả ra màn hình
        if ket_qua[0] == 1 or ket_qua[0] == "Canceled": 
            st.error("⚠️ CẢNH BÁO: Khách hàng này có khả năng CAO sẽ HỦY PHÒNG!")
        else:
            st.success("✅ YÊN TÂM: Khách hàng này khả năng cao sẽ KHÔNG HỦY PHÒNG.")
            
    except FileNotFoundError:
        st.warning("⚠️ Chưa tìm thấy file 'hotel_model.pkl'.")