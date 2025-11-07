# 🍽️ Predicting Estimated Delivery Time for Food Orders  

---

## 📘 Project Overview  

In an age of instant gratification, **accurate food delivery time prediction** has become vital to customer satisfaction and operational success.  
This project develops and compares several **machine learning models** to predict **Estimated Delivery Time (ETA)** based on features like distance, driver ratings, weather, traffic, and vehicle condition.  
The final models empower businesses to optimize routes, allocate resources efficiently, and improve customer experience.  

---

## 🎯 Problem Statement  

The objective is to create robust predictive models that estimate delivery time using internal and external factors such as:  
- Driver demographics (age, rating, workload)  
- Environmental conditions (weather, traffic, festival)  
- Order and location details (distance, vehicle type, order type)  

These predictions help food delivery platforms to:  
- Improve ETA accuracy  
- Enhance delivery partner utilization  
- Increase transparency and customer satisfaction  

---

## 📊 Dataset Summary  

- **Source:** Kaggle — Food Delivery Dataset (India)  
- **Target Variable:** `Time_Taken` (minutes)  
- **Features:**
  - **Delivery Person Info:** Age, Ratings, Multiple Deliveries, Vehicle Condition, Type of Vehicle  
  - **Order & Timing:** Order Time, Pickup Time, Type of Order  
  - **Location:** Restaurant and Delivery coordinates → calculated `Delivery_Distance_km`  
  - **External Factors:** Weather, Traffic Density, Festival, City Type  

---

## 🧹 Data Preprocessing  

1. Removed text artifacts (“(min)”, “conditions”).  
2. Converted all datetime and numeric fields.  
3. Imputed missing values (mode for categorical features like `Festival` and `City`).  
4. Created new derived columns:
   - `Delivery_Distance_km` (via geodesic distance)  
   - `Order_to_Pickup_Duration` (in minutes)
5. Applied:
   - **One-Hot Encoding** for nominal categories  
   - **Ordinal Encoding** for ordered features  
   - **StandardScaler** for normalization  
6. Train–test split: **80/20** with `random_state=42`.  

---

## 🔍 Exploratory Data Analysis  

### Highlights
- Most delivery partners are aged **20–40** years.  
- Ratings skew high (4–5 range).  
- Delivery distances range from **2.5 km – 20 km**.  
- Bicycles → shorter delivery times;  
  Festivals → longer times.  
- Higher traffic and poor vehicle condition increase ETA.  
- Ratings and vehicle condition negatively correlate with delivery time.  

---

## 🧠 Machine Learning Models  

### 1️⃣ K-Nearest Neighbors (KNN)

**Algorithm:** Instance-based regression using distance weighting  

**Hyperparameters:**
- `n_neighbors = 15`
- `weights = 'distance'`
- `metric = 'minkowski'`  

**Performance Metrics:**

| Metric | Value |
|:-------|:------|
| **MAE** | 5.05 minutes |
| **RMSE** | 3.96 minutes |
| **R²** | 0.7097 |
| **Adjusted R²** | 0.7090 |

➡️ The model captures general patterns but struggles with outliers and local variance.  

---

### 2️⃣ Decision Tree Regressor  

**Algorithm:** Tree-based model splitting data by feature thresholds.  

**Hyperparameters:**
- `random_state = 0`
- `max_depth = None`
- `min_samples_split = 2`

**Performance Metrics:**

| Metric | Value |
|:-------|:------|
| **MAE** | 3.22 minutes |
| **RMSE** | 4.06 minutes |
| **R²** | 0.8125 |
| **Adjusted R²** | 0.8120 |

➡️ Good fit with some overfitting tendencies due to unconstrained depth.  

---

### 3️⃣ Random Forest Regressor  

**Algorithm:** Ensemble of multiple decision trees (bagging).  

**Hyperparameters:**
- `n_estimators = 100`
- `random_state = 42`

**Performance Metrics:**

| Metric | Value |
|:-------|:------|
| **MAE** | 3.08 minutes |
| **RMSE** | 3.85 minutes |
| **R²** | 0.830 |
| **Adjusted R²** | 0.8304 |

➡️ Excellent predictive accuracy with minimal overfitting.  

---

### 4️⃣ SVR + MLP Stacked Pipeline  

**Pipeline Components**
- **Base learners:**
  - SVR (RBF kernel)  
  - MLP (2 hidden layers, ReLU activation)  
- **Meta-learner:** RidgeCV (with passthrough)  

**Tuning:**
- SVR: `C∈[1e-2,1e3]`, `γ∈[1e-4,1]`, `ε∈{0.1, 0.2, 0.3}`  
- MLP: hidden layers (64, 32) or (128, 64), α∈[1e-4,1e-2], batch = 32  

**Performance Metrics:**

| Model | R² | RMSE (min) | MAE (min) |
|:------|----:|------------:|-----------:|
| SVR | 0.758 | — | — |
| MLP | 0.777 | — | — |
| **Stacked Ensemble** | **0.803** | **4.12** | **3.26** |

➡️ Strong generalization and smoother prediction curve.  

---

### 5️⃣ XGBoost Regressor  

**Algorithm:** Gradient boosting with regularization  

**Hyperparameters:**
- `objective = 'reg:squarederror'`
- `max_depth = 6`
- `learning_rate = 0.1`
- `n_estimators = 100`
- `eval_metric = 'rmse'`

**Performance Metrics:**

| Metric | Value |
|:-------|:------|
| **MAE** | 3.10 minutes |
| **RMSE** | 3.84 minutes |
| **R²** | 0.8317 |
| **Adjusted R²** | 0.8303 |

➡️ **Best overall performer**, balancing accuracy, interpretability, and generalization.  

**Top Influential Features:**
1. Multiple Deliveries  
2. Delivery Person Ratings  
3. Sunny Weather  
4. Traffic Density  
5. Vehicle Condition  
6. Festival Day  
7. Delivery Distance  
8. Driver Age  

---

## ⚖️ Comparative Model Performance  

| Model | MAE (min) | RMSE (min) | R² | Adj. R² |
|:------|-----------:|-----------:|----:|---------:|
| **KNN** | 5.05 | 3.96 | 0.7097 | 0.7090 |
| **Decision Tree** | 3.22 | 4.06 | 0.8125 | 0.8120 |
| **Random Forest** | 3.08 | 3.85 | 0.830 | 0.8304 |
| **SVR + MLP Stack** | 3.26 | 4.12 | 0.803 | — |
| **XGBoost (Best)** | **3.10** | **3.84** | **0.8317** | **0.8303** |

✅ **XGBoost** delivered the most reliable and consistent predictions with the lowest error metrics.  

---

## 💡 Insights & Recommendations  

**Key Insights**
- Assigning **multiple deliveries** increases ETA.  
- **Higher driver ratings** → faster deliveries.  
- **Adverse weather** and **traffic congestion** cause major delays.  
- **Festivals** and **longer distances** raise delivery time.  
- **Vehicle maintenance** directly impacts efficiency.  

**Recommendations**
1. Limit simultaneous deliveries per rider.  
2. Assign experienced drivers during peak hours.  
3. Integrate real-time traffic + weather APIs for ETA adjustments.  
4. Introduce delivery-time buffers on festivals.  
5. Match route length with suitable vehicle condition/type.  

---

## 🧰 Tech Stack  

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `geopy`, `mlxtend`, `keras`  
- **Tools:** Jupyter Notebook  

---

## 🚀 How to Run  

```bash
git clone <repo-url>
cd food-delivery-eta-prediction
pip install -r requirements.txt
jupyter notebook
