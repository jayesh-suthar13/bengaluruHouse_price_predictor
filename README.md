# Bengaluru House Price Prediction

This project predicts house prices in Bengaluru using regression models based on features like location, area, number of bedrooms, and bathrooms.

## Workflow

1. **Data Cleaning** – Removed null values and inconsistent entries
2. **Feature Engineering** – Processed features like `location`, `BHK`, `bath`, and `total_sqft`
3. **Outlier Removal** – Eliminated extreme and unrealistic values
4. **Modeling** – Built and compared:

   * Linear Regression
   * Lasso Regression
   * Ridge Regression
5. **Preprocessing** – Used `OneHotEncoder`, `StandardScaler`, and `ColumnTransformer`
6. **Evaluation** – Used R² scores:

   ```
   Linear Regression: -4.63e+18
   Lasso: 0.813
   Ridge: 0.812
   ```
7. **Deployment** – Final Ridge model deployed using Streamlit

## Tools Used

* Python, Pandas, scikit-learn, matplotlib , seaborn, pipeline
* Streamlit for web deployment
* Pickle for model serialization

## How to Run

```bash
pip install -r requirements.txt
streamlit run app2.py
```

## Files

* `app2.py`: Streamlit app
* `RidgeModel.pkl`: Trained model
* `columns.pkl`: Feature metadata

THANK YOU!
