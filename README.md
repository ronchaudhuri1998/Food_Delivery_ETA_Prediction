# Food_Delivery_ETA_Prediction
A machine learning pipeline that predicts food delivery times using engineered features (geodesic distance, weather, traffic, and cyclical order timestamps). Multiple regression models—including KNN, Decision Tree, Random Forest, XGBoost, SVR, and MLP have been used to check which model predicts the delivery time most accurately. 

- Engineered features including geodesic distance, order time cycled into sine/cosine, weather, traffic density, and categorical encodings; conducted EDA to understand drivers of delivery time.

- Built and tuned six regression models (KNN, Decision Tree, Random Forest, XGBoost, SVR, MLP) using RandomizedSearchCV and HalvingRandomSearchCV with k-fold CV.

- Deployed an XGBoost model that achieved R² = 0.832, RMSE = 3.84 min and a stacked SVR+MLP ensemble with R² = 0.803, RMSE = 4.12 min on hold-out data.
