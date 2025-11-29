# from components.model_evaluate import ModelEvaluator

# ev = ModelEvaluator(
#     project_id="ml-ai-portfolio",
#     dataset_id="taxi_forecasting"
# )

# result = ev.evaluate(
#     model_path="ml-ai-portfolio.taxi_forecasting.daily_arima_model_v2")

# print("Evaluation:", result)



# 'rmse': 21180.476079191998, 
# 'mae': 14653.40960948927,
# 'mape': 16.088306500097218, 
# 'rows': 30


# -----------------------------------------------


from components.model_evaluate import ModelEvaluator

ev = ModelEvaluator(
    project_id="ml-ai-portfolio",
    dataset_id="taxi_forecasting"
)

# Model 1: Default ARIMA
result = ev.evaluate(
    model_path="ml-ai-portfolio.taxi_forecasting.daily_arima_model_v2"
)

print("Default ARIMA metrics:")
print(result)


# 'rmse': 21180.476079191998, 
# 'mae': 14653.40960948927, 
# 'mape': 16.088306500097218, 
# 'rows': 30}