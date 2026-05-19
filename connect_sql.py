import pyodbc
import pandas as pd

try:
    # KẾT NỐI SQL SERVER
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=HotelReservationDB;'
        'Trusted_Connection=yes;'
    )

    print("Kết nối SQL Server thành công!\n")

    # ==============================
    # 1. LẤY DỮ LIỆU GỐC
    # ==============================
    query = """
    SELECT TOP 5 *
    FROM Hotel_Reservations
    """

    df = pd.read_sql(query, conn)

    print("Dữ liệu gốc:")
    print(df)


    # ==============================
    # 2. FEATURE ENGINEERING
    # ==============================
    query_feature = """
    SELECT 
        Booking_ID,
        no_of_adults,
        no_of_children,
        (no_of_adults + no_of_children) AS total_people,
        no_of_weekend_nights,
        no_of_week_nights,
        (no_of_weekend_nights + no_of_week_nights) AS total_nights,
        lead_time,
        avg_price_per_room,
        booking_status
    FROM Hotel_Reservations
    """

    df_feature = pd.read_sql(query_feature, conn)

    print("\nFeature Engineering:")
    print(df_feature.head())


    # ==============================
    # 3. PHÂN TÍCH BOOKING STATUS
    # ==============================
    query_analysis = """
    SELECT 
        booking_status,
        COUNT(*) AS total
    FROM Hotel_Reservations
    GROUP BY booking_status
    """

    df_analysis = pd.read_sql(query_analysis, conn)

    print("\nPhân tích booking status:")
    print(df_analysis)


    # ==============================
    # 4. EXPORT FILE CSV
    # ==============================
    df_feature.to_csv("hotel_feature_data.csv", index=False)

    print("\nĐã export file: hotel_feature_data.csv")


    # CLOSE CONNECTION
    conn.close()


except Exception as e:
    print("Lỗi xảy ra:")
    print(e)