import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# -------------------------- Caching Functions --------------------------
@st.cache_data
def load_data():
    df_encoded = pd.read_csv(r"D:\\Swiggy\\data\\encoded_data.csv")
    df_clustered = pd.read_csv(r"D:\\Swiggy\\data\\clustered_data.csv")
    cleaned_df = pd.read_csv(r"D:\\Swiggy\\data\\cleaned_data.csv")

    cleaned_df['cluster'] = df_clustered['cluster']
    cleaned_df['cuisine'] = cleaned_df['cuisine'].fillna("").apply(lambda x: ', '.join(sorted(set(i.strip() for i in x.split(',')))))

    # Explode for dropdown
    df_multi = cleaned_df.copy()
    df_multi['cuisine'] = df_multi['cuisine'].str.split(',')
    df_multi = df_multi.explode('cuisine')
    df_multi['cuisine'] = df_multi['cuisine'].str.strip()

    return cleaned_df, df_multi

@st.cache_resource
def load_models():
    with open(r"D:\\Swiggy\\data\\encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)
    with open(r"D:\\Swiggy\\data\\scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(r"D:\\Swiggy\\data\\kmeans_model.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    return encoder, scaler, kmeans

# -------------------------- UI Setup --------------------------
st.set_page_config("Swiggy Recommender", layout="wide")

st.markdown("""
    <style>
        .stButton > button {
            background-color: #8e44ad;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .css-1d391kg, .css-1cpxqw2 {
            background-color: #1e1e2f;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(to right, #8e44ad, #3498db); 
                color: white; border-radius: 12px; margin-bottom: 25px;">
        <h1>🍽️ Swiggy Restaurant Recommender</h1>
        <p style="font-size: 18px;">Find top-rated restaurants based on your preferences! 🔍</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------- Load Data/Models --------------------------
cleaned_df, df_multi = load_data()
encoder, scaler, kmeans = load_models()

all_cities = sorted({col.split('_')[1] for col in encoder if col.startswith('city_')})
all_cuisines = df_multi['cuisine'].drop_duplicates().sort_values().tolist()

# -------------------------- Sidebar --------------------------
st.sidebar.header("🎯 Customize Your Search")
city = st.sidebar.selectbox("📍 Choose your city", all_cities)
cuisine = st.sidebar.selectbox("🍱 Preferred cuisine", all_cuisines)
rating = st.sidebar.slider("⭐ Minimum rating", 1.0, 5.0, 4.0, 0.1)
cost = st.sidebar.number_input("💰 Budget (₹)", 50, 1000, 300, 50)
top_n = st.sidebar.slider("🔝 Number of results", 1, 10, 5)

# -------------------------- Recommendation Logic --------------------------
def recommend(city, cuisine, rating, cost, top_n):
    # Create input with same columns as encoder
    input_df = pd.DataFrame(columns=scaler.feature_names_in_)
    input_df.loc[0] = 0  # fill all with zero

    # Fill numeric fields
    if 'rating' in input_df.columns:
        input_df.at[0, 'rating'] = rating
    if 'rating_count' in input_df.columns:
        input_df.at[0, 'rating_count'] = 100
    if 'cost' in input_df.columns:
        input_df.at[0, 'cost'] = cost

    # Set one-hot encoded values
    if f'city_{city}' in input_df.columns:
        input_df.at[0, f'city_{city}'] = 1
    if f'cuisine_{cuisine}' in input_df.columns:
        input_df.at[0, f'cuisine_{cuisine}'] = 1

    X_scaled = scaler.transform(input_df)
    cluster_label = kmeans.predict(X_scaled)[0]

    filtered = df_multi[(df_multi['cluster'] == cluster_label) &
                        (df_multi['cuisine'].str.lower() == cuisine.lower()) &
                        (df_multi['city'].str.contains(city))]

    return filtered.drop_duplicates('name').sort_values(by='rating', ascending=False).head(top_n)

# -------------------------- Show Recommendations --------------------------
if st.sidebar.button("🔍 Show Recommendations"):
    with st.spinner("Fetching delicious matches... 🍲"):
        time.sleep(1)
        results = recommend(city, cuisine, rating, cost, top_n)

    if results.empty:
        st.warning("No matches found. Try changing filters.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"""
                <div style="background-color: #a367b8; padding: 16px; border-radius: 10px;
                     box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; color: #000000;">
                    <h4 style="margin-bottom: 5px;">🍴 {row['name']}</h4>
                    <p style="margin: 0;">
                        📍 {row['city']}<br>
                        🍱 Cuisine: {row['cuisine']}<br>
                        ⭐ Rating: {row['rating']} | 💸 ₹{row['cost']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.caption("🍲 Built with ❤️ using Streamlit | Powered by KMeans | © 2025")
