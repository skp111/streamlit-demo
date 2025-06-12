import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load saved models
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("Customer Segmentation & Business Recommendations")
st.markdown("Enter customer information to predict segment and get personalized insights.")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
work_experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3)

if st.button("Predict Segment"):
    # Prepare input
    input_data = pd.DataFrame([[age, work_experience, family_size]],
                              columns=["Age", "Work_Experience", "Family_Size"])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.subheader(f"Predicted Customer Segment: Cluster {cluster}")

    # Define profiling and business recommendations
    profiles = {
        0: ("Young professionals", "Promote career growth tools and premium financial planning services."),
        1: ("Mid-age stable earners", "Offer family bundles, housing loans, or investment plans."),
        2: ("Large families with experience", "Provide cost-effective products, loyalty offers, and educational plans."),
        3: ("Retired or late career", "Focus on health insurance, leisure travel, and estate planning."),
        4: ("Early career starters", "Introduce starter packages, educational loans, or gig-economy services.")
    }

    profile, recommendation = profiles.get(cluster, ("Unknown", "No recommendation available."))

    st.markdown(f"**Customer Profile:** {profile}")
    st.markdown(f"**Business Recommendation:** {recommendation}")

    # Generate dummy cluster centers for visualization (if real centers unavailable)
    try:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_df = pd.DataFrame(centers, columns=["Age", "Work_Experience", "Family_Size"])
        cluster_df["Cluster"] = [f"Cluster {i}" for i in range(len(cluster_df))]
        cluster_df["Color"] = ["highlight" if i == cluster else "normal" for i in range(len(cluster_df))]

        # Add user point
        user_point = pd.DataFrame({
            "Age": [age],
            "Work_Experience": [work_experience],
            "Family_Size": [family_size],
            "Cluster": ["Your Input"],
            "Color": ["user"]
        })

        plot_df = pd.concat([cluster_df, user_point], ignore_index=True)

        fig = px.scatter_3d(plot_df, x="Age", y="Work_Experience", z="Family_Size", 
                            color="Color", symbol="Cluster",
                            title="Cluster Centers and Your Input",
                            color_discrete_map={"highlight": "red", "normal": "gray", "user": "blue"})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")
