# Demand-Forecasting-using-Machine-learning and Deep Learning
This project aims to predict the demand (sales) for various products sold in multiple stores across different time periods.<br>
Contributors : Pooja and Nithish<br>
**Project Overview**
This project aims to predict the demand (sales) for various products sold in multiple stores across different time periods. By leveraging historical sales data and applying advanced machine learning and DL techniques, the project intends to forecast the stock requirements for products, helping businesses optimize their inventory and avoid overstocking or understocking situations.

The project includes various stages such as data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and deployment of the best-performing model.<br>
<br> **RandomForestRegressor vs LSTM**

- The features used to build the models for demand forecasting were: store, item, month, day, weekday, holidays, m1, m2, weekend, sales_lag_1, sales_lag_2, and sales_lag_3.
- As demand forecasting is a type of time series forecasting, the data was divided sequentially to ensure the temporal order was maintained.
- Maintaining the sequence is crucial because time series data often exhibit trends, seasonality, or other time-dependent patterns that could be disrupted by random shuffling.
- By preserving the temporal order, the model can learn from past data behavior and make accurate predictions about future demand.
- The data was prepared for training a Long Short-Term Memory (LSTM) model by creating input sequences (X) and corresponding targets (y).
- This setup was designed to predict daily sales by analyzing patterns from the past 30 days of data.
- For the RandomForestRegressor model, *RandomizedSearchCV* was used to find the best hyperparameters, optimizing the model’s performance.
- Using the above feature engineering, the RandomForestRegressor model outperformed the LSTM model for this dataset.

