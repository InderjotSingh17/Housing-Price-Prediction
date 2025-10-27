# 🏡Housing Price Prediction

This project predicts house prices in California using Machine Learning.  
It is built using **Python, Pandas, Scikit-Learn** and a **Random Forest** model.

---

## 📌 Project Summary

- **Problem type:** Regression  
- **Dataset:** California Housing (Kaggle)  
- **Target variable:** `median_house_value`  
- **Main model used:** `RandomForestRegressor` (baseline)  
- **Evaluation metrics:** MAE, RMSE, R²

---

## 📈 Final Model Performance (from this run)

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | **31,639.37** |
| Mean Squared Error (MSE) | **2,404,745,975.12** |
| Root Mean Squared Error (RMSE) | **49,038.21** |
| R² Score | **0.8165** |

> These values reflect the model and preprocessing choices used in this notebook (median imputation for `total_bedrooms`, one-hot encoding for `ocean_proximity`, train/test split with `random_state=42`, and RandomForest with `n_estimators=100`).

---

## 🧹 Data Preprocessing Performed

- Checked and handled missing values (filled `total_bedrooms` with median)  
- One-hot encoded `ocean_proximity` (categorical)  
- Train/Test split: 80% train / 20% test (`random_state=42`)  
- (For RandomForest scaling is not required; for other models StandardScaler was used where appropriate)

---

## 🤖 Model (example snippet)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## 🔍 Steps Performed

1. Loaded the California Housing dataset
2. Checked dataset info and statistics
3. Handled missing values in `total_bedrooms`
4. Train-test split (80% train, 20% test)
5. Feature scaling using StandardScaler
6. Trained a Linear Regression model
7. Evaluated results using RMSE and R² score

---

## 💡 Future Improvements
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Try gradient boosting (XGBoost / LightGBM / CatBoost)

Add engineered features: rooms_per_household, bedrooms_per_room, population_per_household

Cross-validation and target transformation (log) if needed

Deploy model with Streamlit / Flask (use saved joblib file)

---

## 👨‍💻 Author
**Inderjot Singh**  
B.Tech CSE (AI & ML)  

GitHub: https://github.com/InderjotSingh17  

---

## ⭐ Show Support
If you like this project, give it a ⭐ on GitHub.

---


