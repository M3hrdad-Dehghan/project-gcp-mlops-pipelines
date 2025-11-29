from components.data_loader import DataLoader

loader = DataLoader(
    project_id="ml-ai-portfolio",
    dataset_id="taxi_forecasting",
    table_id="aggregated_daily_2022",
)


# 2) داده کامل را از BigQuery می‌خوانیم
df = loader.load_data()
print("[Script] Full data shape:", df.shape)

# 3) split train/test بر اساس تاریخ
train_df, test_df = loader.train_test_split(
    df=df,
    train_end_date="2022-10-31",  # اینجا cutoff رو تعریف می‌کنیم
)


print("[Script] Train:", train_df["trip_date"].min(), "->", train_df["trip_date"].max(), "rows:", len(train_df))
print("[Script] Test :", test_df["trip_date"].min(), "->", test_df["trip_date"].max(), "rows:", len(test_df))



# 4) ذخیره‌ی train در جدول جدا
loader.save_to_bigquery(
    df=train_df,
    target_table="train_daily_2022",
    write_disposition="WRITE_TRUNCATE",
)


# 5) ذخیره‌ی test در جدول جدا
loader.save_to_bigquery(
    df=test_df,
    target_table="test_daily_2022",
    write_disposition="WRITE_TRUNCATE",
)


print("[Script] Done creating train_daily_2022 and test_daily_2022 tables.")
