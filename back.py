import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import joblib
from tensorflow.keras.preprocessing import image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load trained CNN model for food recognition
print("Loading CNN model...")
model = tf.keras.models.load_model("smartspoon.keras")  

# Load Excel dataset for recommendations
print("Loading recommendation dataset...")
df = pd.read_excel("recomf.11.xlsx")

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
def load_sentiment_system():
    try:
        print("Loading complete sentiment analysis system...")
        system = joblib.load('improved_sentiment_analysis_system.pkl')
        return system
    except Exception as e:
        print(f"Error loading sentiment system: {e}")
        return None

# Initialize the sentiment system
sentiment_system = load_sentiment_system()

def predict_food_from_image(image_path):
    try:
        # Strip quotes if present
        image_path = image_path.strip('"\'')
        
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model(img_array, training=False)
        predicted_class_index = np.argmax(predictions.numpy()[0])

        predicted_food = class_labels.get(predicted_class_index, "Unknown Food")
        return predicted_food
    except Exception as e:
        print(f"❌ Error in image processing: {e}")
        return "Error"

def get_nutrition_data(food_name, api_key):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {'query': food_name, 'api_key': api_key}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return "❌ API Error: Invalid response"
        data = response.json()
        if 'foods' in data and data['foods']:
            return {n['nutrientName']: n.get('value', 'N/A') for n in data['foods'][0].get('foodNutrients', [])}
        return "❌ Nutrition data not found"
    except Exception as e:
        return f"❌ API Error: {e}"

def recommend_alternative(nutrition_data):
    try:
        sodium = nutrition_data.get("Sodium, Na")
        if sodium is None:
            return "⚠️ No sodium data available.", []

        sodium = float(sodium)
        HIGH_SODIUM_THRESHOLD = 400
        LOW_SODIUM_THRESHOLD = 100

        if sodium > HIGH_SODIUM_THRESHOLD:
            sodium_status = "⚠️ High sodium! Try these alternatives"
            df_filtered = df[df['sodium_mg'] <= LOW_SODIUM_THRESHOLD]
        elif sodium < LOW_SODIUM_THRESHOLD:
            sodium_status = "⚠️ Low sodium! Try these alternatives"
            df_filtered = df[df['sodium_mg'] >= HIGH_SODIUM_THRESHOLD]
        else:
            sodium_status = "✅ Healthy sodium level! You can also try these"
            df_filtered = df.iloc[(df['sodium_mg'] - sodium).abs().argsort()[:5]]

        recommended_foods = df_filtered['food_name'].tolist()[:5]
        return sodium_status, recommended_foods
    except Exception as e:
        return f"❌ Error in recommendation: {e}", []

def analyze_sentiment(review_text):
    # If sentiment system is loaded, use it
    if sentiment_system is not None:
        try:
            # Use the system's predict function which handles everything
            sentiment, _, _ = sentiment_system['predict_function'](
                review_text, 
                sentiment_system['model'],
                sentiment_system['label_encoder']
            )
            return sentiment
        except Exception as e:
            print(f"Warning: System prediction failed: {e}")
            # Fall back to VADER
    
    # Fallback to VADER if system loading failed or prediction failed
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review_text)
    
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def find_similar_reviews(review_text, sentiment):
    if sentiment_system is None:
        # Return example reviews if no dataset
        examples = {
            'Positive': [
                "The food was absolutely delicious and had the perfect amount of flavor.",
                "I loved every bite of this meal, it was well balanced and tasty.",
                "The freshness of the ingredients really came through in this dish.",
                "Amazing taste and the portion size was perfect.",
                "One of the best meals I've had, flavorful without being too salty."
            ],
            'Negative': [
                "The dish was way too salty and I couldn't finish it.",
                "I didn't enjoy this meal at all, it lacked flavor and was poorly prepared.",
                "The food was too greasy and made me feel unwell afterward.",
                "Extremely disappointing, wouldn't recommend this to anyone.",
                "The dish was bland and tasteless despite being advertised as flavorful."
            ],
            'Neutral': [
                "The food was okay, nothing special but not bad either.",
                "It was a decent meal but I wouldn't go out of my way to have it again.",
                "Average taste, it did the job but didn't impress me.",
                "The dish was acceptable but could use more seasoning.",
                "Neither good nor bad, just a standard meal without much character."
            ]
        }
        return examples.get(sentiment, examples['Neutral'])
    
    try:
        # Load reviews dataset
        reviews_df = pd.read_excel('reviews.xlsx')
        
        # Filter by sentiment
        sentiment_reviews = reviews_df[reviews_df['Sentiment'] == sentiment]
        
        # If no reviews with that sentiment, return examples
        if len(sentiment_reviews) == 0:
            return find_similar_reviews(review_text, sentiment)
            
        # Process the reviews using the same preprocessing function
        processed_reviews = sentiment_reviews['Response'].apply(
            sentiment_system['preprocess_function']
        )
        
        # Simple keyword matching for similarity (as a fallback)
        review_words = set(review_text.lower().split())
        similarity_scores = []
        
        for idx, review in enumerate(processed_reviews):
            review_set = set(review.lower().split())
            common_words = review_words.intersection(review_set)
            similarity = len(common_words) / max(len(review_words), len(review_set), 1)
            similarity_scores.append((idx, similarity))
        
        # Get top 5 most similar reviews
        top_indices = [idx for idx, score in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]]
        return sentiment_reviews.iloc[top_indices]['Response'].tolist()
    
    except Exception as e:
        print(f"Error finding similar reviews: {e}")
        # Return example reviews if anything fails
        examples = {
            'Positive': [
                "The food was absolutely delicious and had the perfect amount of flavor.",
                "I loved every bite of this meal, it was well balanced and tasty.",
                "The freshness of the ingredients really came through in this dish.",
                "Amazing taste and the portion size was perfect.",
                "One of the best meals I've had, flavorful without being too salty."
            ],
            'Negative': [
                "The dish was way too salty and I couldn't finish it.",
                "I didn't enjoy this meal at all, it lacked flavor and was poorly prepared.",
                "The food was too greasy and made me feel unwell afterward.",
                "Extremely disappointing, wouldn't recommend this to anyone.",
                "The dish was bland and tasteless despite being advertised as flavorful."
            ],
            'Neutral': [
                "The food was okay, nothing special but not bad either.",
                "It was a decent meal but I wouldn't go out of my way to have it again.",
                "Average taste, it did the job but didn't impress me.",
                "The dish was acceptable but could use more seasoning.",
                "Neither good nor bad, just a standard meal without much character."
            ]
        }
        return examples.get(sentiment, examples['Neutral'])

# Main function to run the program
def main():
    api_key = "u7bvXwyQMdTaLmMGz1Cr72JZMucTr5rjGEPEbhsi"
    image_path = input("Enter the image path (default: C:\\Users\\91910\\Downloads\\biriyani.jpg): ") or r"C:\Users\91910\Downloads\biriyani.jpg"
    
    # Predict food from image
    predicted_food = predict_food_from_image(image_path)
    print(f"✅ Predicted Food: {predicted_food}")
    
    if predicted_food != "Error" and predicted_food != "Unknown Food":
        # Get nutrition data
        nutrition_data = get_nutrition_data(predicted_food, api_key)
        print(f"📊 Nutrition Data: {nutrition_data}")
        
        # Get recommendations
        sodium_status, recommendations = recommend_alternative(nutrition_data)
        print(f"{sodium_status}: {', '.join(recommendations)}")
        
        # Ask for user review
        user_review = input("💬 How was the food? ")
        
        # Analyze sentiment
        sentiment = analyze_sentiment(user_review)
        print(f"🔹 Sentiment Analysis: {sentiment}")
        
        # Find similar reviews
        similar_reviews = find_similar_reviews(user_review, sentiment)
        print("🔍 Similar Reviews:")
        for review in similar_reviews:
            print(f"- {review}")

if __name__ == "__main__":
    main()