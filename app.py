import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")


# Import model
st.title("Laptop Price Predictor 💻")
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

# Making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # Brand input
    company = st.selectbox("Brand", df["Company"].unique())

with middle_column:
    # Laptop type
    type = st.selectbox("Type", df["TypeName"].unique())

with right_column:
    # Ram size
    ram = st.selectbox("Ram (in GB)", df["Ram"].unique())

# Making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # Weight input
    weight = st.number_input("Weight of laptop in kg", min_value=0.1)

with middle_column:
    # Touchscreen input
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

with right_column:
    # IPS display input
    ips = st.selectbox("IPS Display", ["No", "Yes"])

# Making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # Screen size input
    Screen_size = st.number_input("Screen Size (in Inches)", min_value=1.0)

with middle_column:
    # Resolution input
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600','2560x1440', '2304x1440'])

with right_column:
    # CPU input
    cpu = st.selectbox("CPU Brand", df["Cpu brand"].unique())

# Making 3 cols left_column, middle_column, right_column
left_column, right_column = st.columns(2)
with left_column:
    # HDD input
    hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])

with right_column:
    # SSD input
    ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# GPU input
gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())

# OS input
os = st.selectbox("OS Type", df["os"].unique())

# Prediction button
if st.button("Predict Price"):
    ppi = None
    # Converting touchscreen and ips to binary
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # Parsing resolution
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split('x')[1])
    
    # Prevent division by zero in PPI calculation
    if Screen_size != 0:
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Screen_size
    else:
        st.warning("Screen size cannot be zero. Please provide a valid screen size.")
        ppi = None

    # Creating query for prediction with 12 features
    
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)  # Ensuring the correct shape for the query

    # Creating a DataFrame with the required columns (same as the training data)
    query_df = pd.DataFrame(query, columns=["Company", "TypeName", "Ram", "Weight", "Touchscreen", "IPS", "PPI", "Cpu brand", "HDD", "SSD", "Gpu brand", "os"])

    # Ensure that all the necessary columns are included in the input DataFrame
    if "ppi" not in query_df.columns:
        query_df["ppi"] = ppi
    if "Ips" not in query_df.columns:
        query_df["Ips"] = ips

    # Make the prediction
    try:
        # Using the pipeline to transform and predict
        predicted_price = np.exp(pipe.predict(query_df)[0])  # Exponentiating as price is in log scale
        st.title(f"The Predicted Price of Laptop = Rs {int(predicted_price)}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
