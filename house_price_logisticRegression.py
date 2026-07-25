"""
House Price Classification using Logistic Regression + Streamlit
------------------------------------------------------------------
Logistic Regression is a CLASSIFICATION algorithm, not a regression
algorithm for continuous values. So instead of predicting an exact
price, this app predicts whether a house is "Expensive" or
"Affordable" based on a price threshold you choose.

Run with:
    pip install streamlit pandas numpy scikit-learn matplotlib
    streamlit run house_price_logistic_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Classification", layout="centered")

st.title("🏠 House Price Classification (Logistic Regression)")
st.write(
    "Logistic Regression predicts **categories**, not exact prices. "
    "This app classifies a house as **Expensive** or **Affordable** "
    "based on a price threshold you set."
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
    area = rng.integers(500, 4000, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 50, n)

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
# 2. Choose price column & threshold to create categories
# ---------------------------------------------------------
st.header("2. Define the Classification Target")

all_columns = df.columns.tolist()

price_col = st.selectbox(
    "Select the price column",
    options=all_columns,
    index=all_columns.index("price") if "price" in all_columns else len(all_columns) - 1,
)

default_threshold = float(df[price_col].median())
threshold = st.slider(
    "Price threshold — houses above this are labeled 'Expensive'",
    float(df[price_col].min()),
    float(df[price_col].max()),
    default_threshold,
)

df["price_category"] = np.where(df[price_col] > threshold, "Expensive", "Affordable")
st.write("Class balance:")
st.bar_chart(df["price_category"].value_counts())

# ---------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------
st.header("3. Choose Features")

feature_candidates = [c for c in all_columns if c != price_col]
feature_cols = st.multiselect(
    "Select feature column(s)",
    options=feature_candidates,
    default=feature_candidates,
)

if len(feature_cols) == 0:
    st.warning("Please select at least one feature column to continue.")
    st.stop()

X = df[feature_cols].select_dtypes(include=[np.number])
y = (df["price_category"] == "Expensive").astype(int)  # 1 = Expensive, 0 = Affordable

if X.shape[1] == 0:
    st.error("Selected features must be numeric. Please choose numeric columns.")
    st.stop()

# ---------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------
st.header("4. Train the Model")

test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# Scale features (helps logistic regression converge & compare coefficients)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Threshold Used", f"{threshold:,.0f}")

# Coefficients
st.subheader("Model Coefficients (standardized features)")
coef_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_[0]}
).sort_values("Coefficient", key=abs, ascending=False)
st.dataframe(coef_df)

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap="Blues")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Affordable", "Expensive"])
ax_cm.set_yticklabels(["Affordable", "Expensive"])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
st.pyplot(fig_cm)

# Classification report
st.subheader("Classification Report")
report = classification_report(
    y_test, y_pred, target_names=["Affordable", "Expensive"], output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose())

# ROC curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ---------------------------------------------------------
# 5. Predict on new input
# ---------------------------------------------------------
st.header("5. Classify a New House")

input_data = {}
for feature in X.columns:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    input_data[feature] = st.slider(f"{feature}", min_val, max_val, mean_val)

if st.button("Classify House"):
    input_df = pd.DataFrame([input_data])[X.columns]
    input_scaled = scaler.transform(input_df)
    pred_class = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]

    label = "Expensive 💰" if pred_class == 1 else "Affordable 🏡"
    st.success(f"Prediction: **{label}**")
    st.write(f"Probability of being Expensive: {pred_proba:.2%}")
