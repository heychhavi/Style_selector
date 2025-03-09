import streamlit as st
import os
from PIL import Image
import json
import uuid
from datetime import datetime
import base64
from io import BytesIO
import requests
from wardrobe_recommender import WardrobeRecommender
import torch

# Initialize wardrobe recommender
recommender = WardrobeRecommender()

# API endpoint
API_URL = "http://localhost:8504"

# Page config
st.set_page_config(
    page_title="AI Fashion Assistant",
    page_icon="👔",
    layout="wide"
)

# Initialize session state
if 'style_profile' not in st.session_state:
    st.session_state.style_profile = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

def analyze_user_images(uploaded_images):
    """Analyze uploaded user images using the API"""
    try:
        # Convert images to base64
        image_urls = []
        
        for img in uploaded_images:
            # Read and process image
            image = Image.open(img)
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_urls.append(f"data:image/jpeg;base64,{img_str}")
        
        # Get style analysis from API
        response = requests.post(f"{API_URL}/analyze", json={"imageUrls": image_urls})
        if response.status_code == 200:
            return response.json()["styleProfile"]
        else:
            st.error("Error analyzing images")
            return None
            
    except Exception as e:
        st.error(f"Error analyzing images: {str(e)}")
        return None

# Main UI
st.title("AI Fashion Assistant 👔")
st.write("Let's analyze your style and create your perfect wardrobe!")

# Image Upload Section
st.header("Step 1: Share Your Style")
uploaded_files = st.file_uploader("Upload photos of your typical style (optional)", 
                                type=['png', 'jpg', 'jpeg'], 
                                accept_multiple_files=True)

if uploaded_files:
    st.success(f"Received {len(uploaded_files)} images")
    cols = st.columns(min(3, len(uploaded_files)))
    for idx, img_file in enumerate(uploaded_files[:3]):
        cols[idx].image(img_file, caption=f"Style Image {idx+1}")
    
    if st.button("Analyze My Style"):
        with st.spinner("Analyzing your style from images..."):
            # Get image analysis from API
            style_profile = analyze_user_images(uploaded_files)
            if style_profile is not None:
                st.session_state.style_profile = style_profile
                st.success("Style analysis complete!")

# Style Preferences Section
st.header("Step 2: Style Preferences")

# Get user preferences
occasion = st.selectbox(
    "What's the occasion?",
    ["Business Meeting", "Party", "Casual Outing", "Date Night", "Wedding", "Other"]
)

if occasion == "Other":
    occasion = st.text_input("Please specify the occasion:")

budget = st.number_input("What's your budget for the outfit?", min_value=50.0, value=300.0, step=50.0)

color_preferences = st.multiselect(
    "Select your preferred colors:",
    ["Black", "White", "Navy", "Gray", "Brown", "Red", "Blue", "Green", "Purple", "Pink", "Gold", "Silver"],
    default=["Black", "Navy", "White"]
)

style_preferences = st.multiselect(
    "Select your preferred styles:",
    ["Professional", "Casual", "Formal", "Trendy", "Classic", "Bohemian", "Streetwear", "Elegant", "Sporty"],
    default=["Professional", "Classic"]
)

if st.button("Get Recommendations"):
    if not occasion:
        st.error("Please select an occasion")
    else:
        with st.spinner("Generating personalized recommendations..."):
            # Get recommendations from wardrobe recommender
            recommendations = recommender.get_outfit_recommendations(
                style_profile=torch.tensor(st.session_state.style_profile) if st.session_state.style_profile else torch.zeros(512),
                occasion=occasion,
                budget=budget,
                preferences={
                    "color_preference": color_preferences,
                    "style_preference": style_preferences
                }
            )
            
            if recommendations:
                st.session_state.recommendations = recommendations
                st.success("Generated your personalized recommendations!")

# Display Recommendations
if st.session_state.recommendations:
    st.header("Your Personalized Recommendations")
    
    for i, outfit in enumerate(st.session_state.recommendations, 1):
        st.subheader(f"Outfit {i} (Total: ${outfit['total_price']:.2f})")
        
        # Create columns for each item in the outfit
        cols = st.columns(len(outfit['items']))
        for col, item in zip(cols, outfit['items']):
            with col:
                st.write(f"**{item['name']}**")
                st.write(f"Category: {item['category']}")
                st.write(f"Color: {item['color']}")
                st.write(f"Price: ${item['price']:.2f}")
                st.write(f"Description: {item['description']}")
                if item.get('purchase_link'):
                    st.write(f"[Buy Now]({item['purchase_link']})")
        st.write("---") 