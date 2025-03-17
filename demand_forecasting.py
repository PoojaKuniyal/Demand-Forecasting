
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, DMatrix, train
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('train.csv')
# Data Preprocessing
df.fillna(0, inplace=True)  
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Advanced Feature Engineering
df['store_item_interaction'] = df['store'] * df['item']
df['lag_1'] = df['sales'].shift(1)
df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
df['rolling_std_7'] = df['sales'].rolling(window=7).std()
df.dropna(inplace=True)

# Handle outliers by capping
q_low, q_high = df['sales'].quantile(0.01), df['sales'].quantile(0.99)
df['sales'] = np.clip(df['sales'], q_low, q_high)

# Define features and target
features = ['store', 'item', 'day_of_week', 'month', 'year', 'week_of_year',
            'is_weekend', 'store_item_interaction', 'lag_1', 'rolling_mean_7', 'rolling_std_7']
X = df[features]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Hyperparameter Tuning with Randomized Search
xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                   n_iter=20, cv=5, scoring='neg_mean_squared_error', verbose=1)

random_search.fit(X_train, y_train)

# Best Model Selection
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Train Best Model with Early Stopping (Compatible with Older XGBoost)
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'eval_metric': 'rmse'
}

watchlist = [(dtrain, 'train'), (dtest, 'eval')]

best_model = train(params, dtrain, num_boost_round=best_params['n_estimators'],
                   evals=watchlist, early_stopping_rounds=10, verbose_eval=True)
# Cross-validation
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation RMSE: {np.sqrt(-cv_scores).mean():.2f}')

# Predictions
y_pred = best_model.predict(dtest)
# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}')
# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


# Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=best_model.get_score(importance_type='weight').values(),
            y=best_model.get_score(importance_type='weight').keys())
plt.title('Feature Importance')
plt.show()