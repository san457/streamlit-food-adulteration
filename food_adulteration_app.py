import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Placeholder ANN black‑box function ---
def predict_adulteration(model_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Dummy prediction: random 0 (pure) or 1 (adulterated).
    Replace with real ANN inference.
    """
    return pd.Series(np.random.randint(0, 2, size=len(df)), index=df.index)

# --- Sidebar workflow navigation ---
st.sidebar.title("🔬 Food Adulteration Detector")
step = st.sidebar.radio(
    "Workflow Step",
    ("1. Data Input", "2. Preprocessing", "3. Prediction", "4. Visualization", "5. Results")
)

# Shared state
if "df" not in st.session_state:
    st.session_state.df = None
if "preds" not in st.session_state:
    st.session_state.preds = None

# 1. Data Input
if step == "1. Data Input":
    st.header("1. Load or Enter Data")
    choice = st.radio("Choose input mode", ("Upload CSV", "Manual Entry"))
    if choice == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"Loaded {len(df)} rows and {df.shape[1]} columns")
            st.dataframe(df.head())
    else:
        st.write("Enter feature values for one sample:")
        # Example features—replace these with your actual attributes
        feature_names = ["pH", "Turbidity", "Density", "SugarContent"]
        inputs = {}
        for f in feature_names:
            inputs[f] = st.number_input(f"{f}", value=0.0, format="%.3f")
        if st.button("Add Sample"):
            df = st.session_state.df or pd.DataFrame(columns=feature_names)
            new = pd.DataFrame([inputs])
            st.session_state.df = pd.concat([df, new], ignore_index=True)
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)

# 2. Preprocessing
elif step == "2. Preprocessing":
    st.header("2. Preprocessing")
    if st.session_state.df is None:
        st.warning("No data: go to Step 1 to load or enter samples.")
    else:
        df = st.session_state.df.copy()
        st.write("Current data preview:")
        st.dataframe(df.head())

        # Example preprocessing options
        do_norm = st.checkbox("Normalize features (min–max)")
        if do_norm:
            df = (df - df.min()) / (df.max() - df.min())
        
        remove_zero = st.checkbox("Drop rows with any zero values")
        if remove_zero:
            df = df[(df != 0).all(axis=1)]

        st.session_state.df = df
        st.success("Preprocessing applied.")
        st.dataframe(df.head())

# 3. Prediction
elif step == "3. Prediction":
    st.header("3. Run ANN Prediction")
    if st.session_state.df is None:
        st.warning("No data: go to Step 1.")
    else:
        model_name = st.selectbox("Select ANN model (black‑box)", ["SimpleDense", "ConvNet", "CustomNet"])
        if st.button("Predict Adulteration"):
            df = st.session_state.df
            preds = predict_adulteration(model_name, df)
            st.session_state.preds = preds
            st.session_state.df = df.assign(Adulterated=preds)
            st.success("Prediction complete.")
            st.dataframe(st.session_state.df)

# 4. Visualization
elif step == "4. Visualization":
    st.header("4. Visualization")
    if st.session_state.preds is None:
        st.warning("No predictions: go to Step 3.")
    else:
        df = st.session_state.df
        chart_type = st.selectbox("Select chart type", ["Bar count", "Histogram of a feature"])
        fig, ax = plt.subplots()
        if chart_type == "Bar count":
            df["Adulterated"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_xticklabels(["Pure (0)", "Adulterated (1)"], rotation=0)
            ax.set_ylabel("Sample count")
        else:
            feature = st.selectbox("Feature to histogram", df.columns.drop("Adulterated"))
            df[feature].plot(kind="hist", bins=20, ax=ax)
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
        st.pyplot(fig)

# 5. Results
else:
    st.header("5. Final Results & Recommendations")
    if st.session_state.preds is None:
        st.warning("No predictions: complete Step 3 first.")
    else:
        df = st.session_state.df
        total = len(df)
        num_bad = int(df["Adulterated"].sum())
        st.markdown(f"**{num_bad}** of **{total}** samples flagged as adulterated.")
        if num_bad > 0:
            st.warning("⚠️ Adulteration detected. Recommend further lab tests and inform the treating physician.")
        else:
            st.success("✅ No adulteration detected. Safe for patient consumption.")
