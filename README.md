# 📈 Cryptocurrency Price Forecasting using SARIMA  

🚀 This project focuses on **time series forecasting of Bitcoin (BTC) prices** using **statistical modeling techniques**.  
The aim is to analyze historical cryptocurrency data, apply transformations, and build a **SARIMA (Seasonal ARIMA)** model to predict future Bitcoin closing prices.  

---

## 📂 Project Structure  
- 📊 `all_currencies.csv` → Dataset containing historical cryptocurrency price data  
- 🐍 `index.py` → Main script for data preprocessing, analysis, model training, and forecasting  

---

## 🛠️ Technologies Used  
- 🐍 Python  
- 🧮 Pandas & NumPy → Data preprocessing  
- 📊 Matplotlib & Seaborn → Data visualization  
- 📉 Statsmodels (SARIMAX, ADFuller, Seasonal Decompose) → Time series modeling  
- 🔬 SciPy → Box-Cox transformation  
- 📏 Scikit-learn → Model evaluation (MSE, RMSE)  

---

## 🔎 Workflow  
1. 📥 Load and preprocess the dataset  
2. 💰 Extract Bitcoin closing prices and resample to monthly frequency  
3. 📉 Perform stationarity checks & seasonal decomposition  
4. 🔄 Apply Box-Cox transformation & differencing  
5. 📊 Plot ACF & PACF to determine SARIMA parameters  
6. 🤖 Train multiple SARIMA models & select the best one using **AIC**  
7. 🔮 Forecast Bitcoin prices for future months  
8. 🖼️ Visualize actual vs. predicted prices  

---

## 📊 Results  
✅ The project successfully builds a **SARIMA forecasting model** for Bitcoin prices.  
✅ Forecasted values follow real price trends, giving insights into short-term movements.  
✅ This model can be extended to **other cryptocurrencies** for broader financial forecasting.  

---

## 📌 Future Enhancements  
- 🤖 Add **LSTM / GRU deep learning models** for comparison  
- 💱 Extend forecasting to **multiple cryptocurrencies**  
- 🌐 Build an interactive **web dashboard** for real-time visualization & forecasting  

---


   git clone https://github.com/your-username/crypto-price-forecasting.git
   cd crypto-price-forecasting
