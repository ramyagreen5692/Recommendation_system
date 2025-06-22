# 🍽️ Swiggy Restaurant Recommendation System

A machine learning-powered restaurant recommender built using Streamlit and deployed on Hugging Face Spaces. Users receive personalized restaurant suggestions based on city, cuisine, rating, and budget preferences.

---

## 🚀 Features

- 📍 City & cuisine-based restaurant filtering
- ⭐ Minimum rating & budget slider for fine-tuned results
- 🤖 K-Means Clustering for grouping similar restaurants
- 💬 Cleaned & preprocessed data using One-Hot Encoding and scaling
- 🌐 Beautiful and responsive Streamlit UI
- 🔄 Real-time recommendations with a loading spinner
- 🗂️ CSV file compressed and auto-extracted at runtime to optimize space

---

## 📊 Dataset

- **Source**: Swiggy data (1.5 lakh+ records)
- **Key Columns**:
  - `city`, `cuisine`, `rating`, `rating_count`, `cost`
- **Preprocessing**:
  - Cleaned `cost`, `rating_count`, and missing values
  - Applied One-Hot Encoding and feature scaling
  - Clustered using K-Means
  - Saved: `encoded_data.zip`, `encoder.pkl`, `scaler.pkl`, `kmeans_model.pkl`

---

## 🧠 Recommendation Logic

1. User inputs: city, cuisine, rating, cost
2. Input is encoded and scaled
3. Cluster predicted using the trained KMeans model
4. Restaurants from the same cluster are filtered
5. Top-N results (sorted by rating) are displayed beautifully

---

## 🖼️ App UI

- 🎯 Sidebar for selecting input preferences
- 🎨 Vibrant result cards with restaurant info
- ⏳ Spinner animation for processing feedback
- 👨‍🍳 Improved dropdown options (deduplicated cuisines)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Pandas, Scikit-learn)
- **Model**: K-Means Clustering
- **Deployment**: Hugging Face Spaces
   [https://huggingface.co/spaces/ramyaanbu56/Recommendation_System](url)
---


