# 📊 Apple Stock Price Prediction (2020-01-02 to 2025-01-17)

This project involves predicting Apple's stock price using two approaches:

* **Model 1:** Deep Learning with LSTM
* **Model 2:** Hybrid Approach combining ARIMA and LSTM

The dataset used spans from **January 2, 2020** to **January 17, 2025**.

---

## 📌 Tasks Performed

### 1. 🧹 Data Preparation

* Loaded historical AAPL stock data.
* Removed missing values.
* Extracted features: `Open`, `High`, `Low`, `Volume` and target: `Close`.
* Normalized features and target using `MinMaxScaler`.
* Created sequences of 60 previous time steps to feed into the LSTM model.
* Split the dataset into training and testing sets (no shuffling).

---

### 2. 📈 Data Visualization & Correlation Analysis

#### 🔹 Histograms to Understand Feature Distributions

Used `seaborn.histplot` to visualize the distribution of all numeric features:

```python
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()
```

#### 🔹 Correlation Heatmap

Used `seaborn.heatmap` to display correlations between numerical features, which helps in feature selection and understanding multicollinearity:

```python
so = df.select_dtypes(include="number").corr()
plt.figure(figsize=(15, 15))
sns.heatmap(so, annot=True)
plt.show()
```

---

## 🤖 Model 1: LSTM (Long Short-Term Memory)

* Built a Sequential LSTM model with 2 stacked LSTM layers followed by a Dense output layer.
* Trained the model for 10 epochs with batch size 32.
* Made predictions on the test set and inversely scaled the predictions.

### 📉 LSTM Model Evaluation:

* **MSE:** 67.15
* **RMSE:** 8.19
* **MAE:** 6.50
* **R²:** 0.90

📊 Plot: Actual vs Predicted AAPL Closing Price using LSTM

---

## 🔀 Model 2: ARIMA + LSTM Hybrid Model

### Step 1: ARIMA Model

* Fitted an ARIMA(1,1,1) model on the `Close` prices.
* Extracted the fitted values and computed residuals.

### Step 2: LSTM on ARIMA Residuals

* Normalized the residuals.
* Built an LSTM model to capture non-linear components left by ARIMA.
* Trained and predicted on residual sequences.

### Step 3: Final Prediction

* Combined ARIMA’s linear trend prediction with LSTM’s nonlinear residual predictions.
* Resulted in significantly more accurate predictions.

### 📉 Hybrid Model Evaluation:

* **MSE:** 9.04
* **RMSE:** 3.01
* **MAE:** 2.20
* **R²:** 0.99

📊 Plot: Actual vs Predicted AAPL Closing Price using ARIMA + LSTM

---

## ✅ Conclusion

| Model        | MSE   | RMSE | MAE  | R²   |
| ------------ | ----- | ---- | ---- | ---- |
| LSTM Only    | 67.15 | 8.19 | 6.50 | 0.90 |
| ARIMA + LSTM | 9.04  | 3.01 | 2.20 | 0.99 |

* The **ARIMA + LSTM hybrid model** dramatically improved prediction accuracy.
* This showcases the power of combining **linear (ARIMA)** and **nonlinear (LSTM)** models for time series forecasting.

---

📁 *Next Step:* Add model saving, loading, and forecasting future prices.
