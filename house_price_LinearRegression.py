"""
House Price Prediction using Linear Regression + Streamlit
------------------------------------------------------------
Run with:
    pip install streamlit pandas numpy scikit-learn matplotlib
    streamlit run house_price_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction (Linear Regression)")
st.write(
    "Upload your own housing dataset (CSV) or use the built-in sample data, "
    "then train a linear regression model and predict prices."
)

# ---------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------
st.header("1. Data")

uploaded_file = st.file_uploader("Upload a CSV file (optional)", type=["csv"])


@st.cache_data
def load_sample_data():
    """Generate a small synthetic housing dataset."""
    rng = np.random.default_rng(42)
    n = 300
    area = rng.integers(500, 4000, n)                # sq ft
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 50, n)                      # years old

    # Synthetic price formula + noise
    price = (
        area * 150
        + bedrooms * 10000
        + bathrooms * 8000
        - age * 500
        + rng.normal(0, 15000, n)
        + 20000
    )

    df = pd.DataFrame(
        {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "price": price.astype(int),
        }
    )
    return df


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded!")
else:
    df = load_sample_data()
    st.info("No file uploaded — using built-in sample dataset.")

st.dataframe(df.head())

# ---------------------------------------------------------
# 2. Feature / target selection
# ---------------------------------------------------------
st.header("2. Choose Features & Target")

all_columns = df.columns.tolist()

target_col = st.selectbox(
    "Select the target column (price)",
    options=all_columns,
    index=all_columns.index("price") if "price" in all_columns else len(all_columns) - 1,
)

feature_cols = st.multiselect(
    "Select feature column(s)",
    options=[c for c in all_columns if c != target_col],
    default=[c for c in all_columns if c != target_col],
)

if len(feature_cols) == 0:
    st.warning("Please select at least one feature column to continue.")
    st.stop()

X = df[feature_cols].select_dtypes(include=[np.number])
y = df[target_col]

if X.shape[1] == 0:
    st.error("Selected features must be numeric. Please choose numeric columns.")
    st.stop()

# ---------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------
st.header("3. Train the Model")

test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:,.2f}")
col2.metric("R² Score", f"{r2:.3f}")

# Coefficients
st.subheader("Model Coefficients")
coef_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_}
)
st.dataframe(coef_df)
st.write(f"Intercept: {model.intercept_:,.2f}")

# Actual vs Predicted plot
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)

# ---------------------------------------------------------
# 4. Predict on new input
# ---------------------------------------------------------
st.header("4. Predict a New House Price")

input_data = {}
for feature in X.columns:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    input_data[feature] = st.slider(
        f"{feature}", min_val, max_val, mean_val
    )

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: {prediction:,.2f}")
