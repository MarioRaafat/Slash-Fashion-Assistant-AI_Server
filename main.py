import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load and preprocess CSV data
def load_csv_data():
    csv_data = {}
    try:
        csv_data['tags'] = pd.read_csv("tags.csv").to_dict(orient="records")
        csv_data['categories'] = pd.read_csv("category.csv").to_dict(orient="records")
        csv_data['colors'] = pd.read_csv("colours.csv").to_dict(orient="records")
        csv_data['brands'] = pd.read_csv("brands.csv").to_dict(orient="records")
        csv_data['products'] = pd.read_csv("products.csv").to_dict(orient="records")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return {}
    return csv_data

# Pass only product IDs to the prompt for validation
def extract_product_ids(products):
    return [product["id"] for product in products]

# Load CSV data into memory
csv_data = load_csv_data()
if not csv_data:
    raise RuntimeError("Failed to load CSV files. Please ensure all files are available and correctly formatted.")

product_ids = extract_product_ids(csv_data["products"])

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Update the prompt
system_instruction = f"""
You are an intelligent assistant for an e-commerce platform. Your responsibilities are:

1. Identify whether the user's input is a casual conversation or a product-related query.
2. For casual conversations, respond with friendly and appropriate replies in JSON format:
   {{
     "intent": "casual_message",
     "response": "Friendly reply to the user's message"
   }}

3. For product-related queries:
   - Analyze the input and extract relevant details such as product type, color, category, or attributes from our database {csv_data}.
   - Match the user's query against the provided list of products: {csv_data["products"]}.
   - Recommend exactly 5 products that match the query, sorted by relevance, in the following JSON format:
   {{
     "intent": "product_search",
     "recommendations": {{
       "colours": [], 
       "materials": [], 
       "categories": [], 
       "styles": [], 
       "brands": [], 
       "tags": []
     }},
     "recommendation_count": 5,
     recommended_products_Ids: [id1, id2, ...],
   }}

4. If the query does not match any products, provide alternatives using the closest attributes (e.g., similar color or category).
5. Ensure all recommendations are based solely on the provided data.
"""

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# Initialize chat session
chat_session = model.start_chat(history=[])

# Start the chat
print("Bot: Hello, how can I help you?")
print()

# Chat loop
while True:
    user_input = input("You: ")
    print()

    # Send the user input to Gemini
    response = chat_session.send_message(user_input)

    # Parse model response
    model_response = response.text

    # Print the response
    print(f'Bot: {model_response}')

    # Append conversation history
    chat_session.history.append({"role": "user", "parts": [user_input]})
    chat_session.history.append({"role": "model", "parts": [model_response]})