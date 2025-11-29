from components.model_trainer import BQMLTrainer

trainer = BQMLTrainer(
    project_id="ml-ai-portfolio",
    dataset_id="taxi_forecasting"
)

model_path = trainer.train_arima_holiday(
    source_table="train_daily_2022",
    model_name="daily_arima_holiday_v2"
)

print("Created ARIMA Holiday model:", model_path)
