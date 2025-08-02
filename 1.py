
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import joblib
import os
import streamlit as st
from tensorflow.keras.preprocessing import image
import nltk
from PIL import Image
import io

# Set page config as the first Streamlit command
st.set_page_config(page_title="SmartSpoon - Food Analyzer", layout="wide")

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Add the class definition for the pickle file to load properly
class ImprovedTextFeatureExtractor:
    """Feature extractor for text classification."""
    def __init__(self):
        pass
    
    def extract_features(self, text):
        """Extract features from text."""
        # This is a placeholder - implement actual feature extraction if needed
        return []

# Add the predict_sentiment function that was referenced in the pickle file
def predict_sentiment(text, model, label_encoder):
    """
    Predict sentiment using the loaded model and label encoder.
    
    Parameters:
    text (str): Text to analyze
    model: The trained sentiment analysis model
    label_encoder: Label encoder for converting numeric predictions to sentiment labels
    
    Returns:
    tuple: (sentiment_label, probability, processed_text)
    """
    try:
        # This is a simple implementation - adjust based on your actual model requirements
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Process text
        processed_text = text.lower()
        
        # Vectorize text
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform([processed_text])
        
        # Predict
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        # Get highest probability
        max_prob = max(prediction_proba[0])
        
        # Convert numeric prediction to label
        sentiment_label = label_encoder.inverse_transform(prediction)[0]
        
        return sentiment_label, max_prob, processed_text
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return "Neutral", 0.5, text

# Food class mapping
class_indices = {  
    0: 'adhirasam', 1: 'aloo_gobi', 2: 'aloo_matar', 3: 'aloo_methi', 4: 'aloo_shimla_mirch',
    5: 'aloo_tikki', 6: 'anarsa', 7: 'ariselu', 8: 'bandar_laddu', 9: 'basundi', 10: 'bhatura',
    11: 'bhindi_masala', 12: 'biryani', 13: 'boondi', 14: 'butter_chicken', 15: 'chak_hao_kheer',
    16: 'cham_cham', 17: 'chana_masala', 18: 'chapati', 19: 'chhena_kheeri', 20: 'chicken_razala',
    21: 'chicken_tikka', 22: 'chicken_tikka_masala', 23: 'chikki', 24: 'daal_baati_churma',
    25: 'daal_puri', 26: 'dal_makhani', 27: 'dal_tadka', 28: 'dharwad_pedha', 29: 'doodhpak',
    30: 'double_ka_meetha', 31: 'dum_aloo', 32: 'gajar_ka_halwa', 33: 'gavvalu', 34: 'ghevar',
    35: 'gulab_jamun', 36: 'imarti', 37: 'jalebi', 38: 'kachori', 39: 'kadai_paneer',
    40: 'kadhi_pakoda', 41: 'kajjikaya', 42: 'kakinada_khaja', 43: 'kalakand', 44: 'karela_bharta',
    45: 'kofta', 46: 'kuzhi_paniyaram', 47: 'lassi', 48: 'ledikeni', 49: 'litti_chokha',
    50: 'lyangcha', 51: 'maach_jhol', 52: 'makki_di_roti_sarson_da_saag', 53: 'malapua',
    54: 'misi_roti', 55: 'misti_doi', 56: 'modak', 57: 'mysore_pak', 58: 'naan',
    59: 'navrattan_korma', 60: 'palak_paneer', 61: 'paneer_butter_masala', 62: 'phirni',
    63: 'pithe', 64: 'poha', 65: 'poornalu', 66: 'pootharekulu', 67: 'qubani_ka_meetha',
    68: 'rabri', 69: 'ras_malai', 70: 'rasgulla', 71: 'sandesh', 72: 'shankarpali',
    73: 'sheer_korma', 74: 'sheera', 75: 'shrikhand', 76: 'sohan_halwa', 77: 'sohan_papdi',
    78: 'sutar_feni', 79: 'unni_appam'
}
class_labels = {k: v.replace("_", " ").title() for k, v in class_indices.items()}

# Function to load the complete sentiment analysis system
@st.cache_resource
def load_sentiment_system():
    try:
        with st.spinner("Loading sentiment analysis system..."):
            system = joblib.load('improved_sentiment_analysis_system.pkl')
        return system
    except Exception as e:
        #st.warning(f"Failed to load sentiment system: {e}. Falling back to VADER.")
        return None

# Load the CNN model
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading food recognition model..."):
            model = tf.keras.models.load_model("smartspoon.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load recommendation dataset
@st.cache_data
def load_recommendations_df():
    try:
        df = pd.read_excel("recomf.11.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
        return pd.DataFrame(columns=['food_name', 'sodium_mg'])

def get_nutrition_data(food_name, api_key):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {'query': food_name, 'api_key': api_key}
    try:
        with st.spinner("Fetching nutrition data..."):
            response = requests.get(url, params=params)
        if response.status_code != 200:
            return "API Error: Invalid response"
        data = response.json()
        if 'foods' in data and data['foods']:
            return {n['nutrientName']: n.get('value', 'N/A') for n in data['foods'][0].get('foodNutrients', [])}
        return "Nutrition data not found"
    except Exception as e:
        return f"API Error: {e}"

def recommend_alternative(nutrition_data, df):
    try:
        sodium = nutrition_data.get("Sodium, Na")
        if sodium is None:
            return "No sodium data available.", []

        sodium = float(sodium)
        HIGH_SODIUM_THRESHOLD = 400
        LOW_SODIUM_THRESHOLD = 100

        if sodium > HIGH_SODIUM_THRESHOLD:
            sodium_status = "High sodium! Try these alternatives"
            df_filtered = df[df['sodium_mg'] <= LOW_SODIUM_THRESHOLD]
        elif sodium < LOW_SODIUM_THRESHOLD:
            sodium_status = "Low sodium! Try these alternatives"
            df_filtered = df[df['sodium_mg'] >= HIGH_SODIUM_THRESHOLD]
        else:
            sodium_status = "Healthy sodium level! You can also try these"
            df_filtered = df.iloc[(df['sodium_mg'] - sodium).abs().argsort()[:5]]

        recommended_foods = df_filtered['food_name'].tolist()[:5]
        return sodium_status, recommended_foods
    except Exception as e:
        return f"Error in recommendation: {e}", []

def analyze_sentiment(review_text, sentiment_system):
    # Always have VADER as a backup
    analyzer = SentimentIntensityAnalyzer()
    
    # If sentiment system is loaded, try to use it first
    if sentiment_system is not None:
        try:
            # Use the system's predict function
            sentiment, _, _ = predict_sentiment(
                review_text, 
                sentiment_system['model'],
                sentiment_system['label_encoder']
            )
            return sentiment
        except Exception as e:
            # If any error occurs, silently fall back to VADER
            pass
    
    # VADER fallback
    scores = analyzer.polarity_scores(review_text)
    
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def find_similar_reviews(review_text, sentiment):
    # Example reviews - Modified to focus on recommendations feedback
    examples = {
        'Positive': [
            "The recommendations were very helpful and I found some great alternatives.",
            "I really liked the recommended options, they were spot on for my taste.",
            "The alternative foods suggested were exactly what I was looking for.",
            "These recommendations helped me find healthier options I wouldn't have considered.",
            "Great recommendations! I tried one and it was delicious."
        ],
        'Negative': [
            "The recommendations didn't match my preferences at all.",
            "I didn't find the suggested alternatives helpful or realistic.",
            "None of the recommended options were available in my area.",
            "The recommendations didn't consider dietary restrictions.",
            "The alternatives suggested were too different from the original food."
        ],
        'Neutral': [
            "The recommendations were okay, some were useful but others weren't.",
            "The alternatives suggested were reasonable but nothing special.",
            "Some recommendations were good, others didn't match my taste.",
            "The recommendations were informative but I'm not sure I'll try them.",
            "The suggested alternatives were interesting but not what I was looking for."
        ]
    }
    
    try:
        # Try to load reviews dataset
        reviews_df = pd.read_excel('reviews.xlsx')
        
        # Filter by sentiment
        sentiment_reviews = reviews_df[reviews_df['Sentiment'] == sentiment]
        
        # If no reviews with that sentiment, return examples
        if len(sentiment_reviews) == 0:
            return examples.get(sentiment, examples['Neutral'])
            
        # Simple keyword matching for similarity
        review_words = set(review_text.lower().split())
        similarity_scores = []
        
        for idx, review in enumerate(sentiment_reviews['Response']):
            review_set = set(str(review).lower().split())
            common_words = review_words.intersection(review_set)
            similarity = len(common_words) / max(len(review_words), len(review_set), 1)
            similarity_scores.append((idx, similarity))
        
        # Get top 5 most similar reviews
        top_indices = [idx for idx, score in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]]
        return sentiment_reviews.iloc[top_indices]['Response'].tolist()
    
    except Exception as e:
        st.warning(f"Error finding similar reviews: {e}. Using example reviews.")
        # Return example reviews if anything fails
        return examples.get(sentiment, examples['Neutral'])

def display_nutrition_data(nutrition_data):
    if isinstance(nutrition_data, dict):
        # Group nutrients into categories
        categories = {
            "Macronutrients": ["Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Energy", "Water"],
            "Vitamins": [n for n in nutrition_data.keys() if "Vitamin" in n],
            "Minerals": ["Calcium, Ca", "Iron, Fe", "Magnesium, Mg", "Phosphorus, P", 
                        "Potassium, K", "Sodium, Na", "Zinc, Zn", "Copper, Cu", "Selenium, Se"],
            "Other": []  # Will catch all others
        }
        
        # Create tabs
        tabs = st.tabs(["Overview"] + list(categories.keys()))
        
        # Overview tab
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Energy", f"{nutrition_data.get('Energy', 'N/A')} kcal")
                st.metric("Protein", f"{nutrition_data.get('Protein', 'N/A')} g")
                st.metric("Carbohydrates", f"{nutrition_data.get('Carbohydrate, by difference', 'N/A')} g")
            
            with col2:
                st.metric("Fat", f"{nutrition_data.get('Total lipid (fat)', 'N/A')} g")
                st.metric("Fiber", f"{nutrition_data.get('Fiber, total dietary', 'N/A')} g")
                st.metric("Sodium", f"{nutrition_data.get('Sodium, Na', 'N/A')} mg")
        
        # Category tabs
        for i, (category, nutrients) in enumerate(categories.items(), 1):
            with tabs[i]:
                if category == "Other":
                    # Add all nutrients not in other categories
                    all_categorized = []
                    for cat_nutrients in list(categories.values())[:-1]:
                        all_categorized.extend(cat_nutrients)
                    
                    nutrients = [n for n in nutrition_data.keys() if n not in all_categorized]
                
                # Display nutrients in this category
                if nutrients:
                    for nutrient in sorted(nutrients):
                        if nutrient in nutrition_data:
                            st.metric(nutrient, nutrition_data[nutrient])
                else:
                    st.info("No data available for this category")
    else:
        st.error(nutrition_data)  # Display error message

# Initialize session states
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    
    if 'predicted_food' not in st.session_state:
        st.session_state.predicted_food = None
    
    if 'nutrition_data' not in st.session_state:
        st.session_state.nutrition_data = None
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    
    if 'sodium_status' not in st.session_state:
        st.session_state.sodium_status = ""
    
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False

# Navigation callback functions
def go_to_results(food, nutrition, sodium_status, recommendations):
    st.session_state.page = 'results'
    st.session_state.predicted_food = food
    st.session_state.nutrition_data = nutrition
    st.session_state.sodium_status = sodium_status
    st.session_state.recommendations = recommendations

def go_to_feedback():
    st.session_state.page = 'feedback'

def go_to_upload():
    st.session_state.page = 'upload'
    st.session_state.predicted_food = None
    st.session_state.nutrition_data = None
    st.session_state.sodium_status = ""
    st.session_state.recommendations = []
    st.session_state.feedback_submitted = False

# Upload page
def show_upload_page():
    st.title("**Smart Food Analyzer**")
    st.write("Upload a food image to get predictions, nutrition info, and recommendations!")
    
    # Load models and data
    model = load_model()
    df = load_recommendations_df()
    
    # API key
    api_key = "u7bvXwyQMdTaLmMGz1Cr72JZMucTr5rjGEPEbhsi"
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"], key="food_image")
    
    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Food"):
            with st.spinner("Analyzing image..."):
                try:
                    # Process the image
                    img = Image.open(uploaded_file).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    predictions = model(img_array, training=False)
                    predicted_class_index = np.argmax(predictions.numpy()[0])
                    predicted_food = class_labels.get(predicted_class_index, "Unknown Food")
                    
                    if predicted_food != "Unknown Food":
                        # Get nutrition data
                        nutrition_data = get_nutrition_data(predicted_food, api_key)
                        
                        # Get recommendations
                        sodium_status, recommendations = recommend_alternative(nutrition_data, df)
                        
                        # Navigate to results page
                        go_to_results(predicted_food, nutrition_data, sodium_status, recommendations)
                    else:
                        st.error("Could not identify the food in this image.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Results page
def show_results_page():
    st.title("**Food Analysis Results**")
    
    # Display predicted food
    st.success(f"**Predicted Food**: {st.session_state.predicted_food}")
    
    # Display nutrition data in expandable section
    with st.expander("📊 View Nutrition Data", expanded=True):
        display_nutrition_data(st.session_state.nutrition_data)
    
    # Display recommendations
    with st.expander("🍽️ Recommendations", expanded=True):
        st.info(f"**{st.session_state.sodium_status}**:")
        if st.session_state.recommendations:
            for i, rec in enumerate(st.session_state.recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("No specific recommendations available.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Analyze Another Food"):
            go_to_upload()
    with col2:
        if st.button("Leave Feedback ➡️"):
            go_to_feedback()

# Feedback page
def show_feedback_page():
    sentiment_system = load_sentiment_system()
    
    st.title("**Feedback on Recommendations**")
    st.write(f"Please share your thoughts on the recommendations for {st.session_state.predicted_food}:")
    
    # User review input
    user_review = st.text_area("What do you think about the recommended alternatives?", height=100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Results"):
            st.session_state.page = 'results'
            st.rerun()  # Changed from st.experimental_rerun()
    
    with col2:
        # Submit button
        if st.button("Submit Feedback"):
            if user_review:
                # Analyze sentiment
                sentiment = analyze_sentiment(user_review, sentiment_system)
                
                # Display sentiment with appropriate color
                if sentiment == "Positive":
                    st.success(f"**Feedback Sentiment**: {sentiment}")
                elif sentiment == "Negative":
                    st.error(f"**Feedback Sentiment**: {sentiment}")
                else:
                    st.info(f"**Feedback Sentiment**: {sentiment}")
                
                # Find similar reviews about recommendations
                similar_reviews = find_similar_reviews(user_review, sentiment)
                
                # Display similar reviews
                with st.expander("🔍 Similar Feedback on Recommendations", expanded=True):
                    for review in similar_reviews:
                        st.write(f"- {review}")
                        
                # Thank user for feedback
                st.success("Thank you for your feedback on our recommendations! This helps us improve our suggestions.")
                
                # Mark feedback as submitted
                st.session_state.feedback_submitted = True
                
                # Show button to analyze another food
                if st.button("Analyze Another Food"):
                    go_to_upload()
            else:
                st.warning("Please enter your feedback before submitting.")

def main():
    # Initialize session state
    init_session_state()
    
    # Show the appropriate page based on state
    if st.session_state.page == 'upload':
        show_upload_page()
    elif st.session_state.page == 'results':
        show_results_page()
    elif st.session_state.page == 'feedback':
        show_feedback_page()

if __name__ == "__main__":
    main()