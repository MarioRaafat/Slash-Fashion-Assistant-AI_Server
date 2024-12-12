import json
import os
import tempfile
import google.generativeai as genai
import utils.csvLoader as csvLoader

csv_data = csvLoader.load_csv_analysis_data()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if not csv_data:
    raise RuntimeError("Failed to load CSV files. Ensure all files are correctly formatted.")

system_instruction = f"""
You are an intelligent assistant specialized in analyzing product images for an e-commerce platform. Your responsibilities are:

1. Analyze the input image and extract relevant features, including but not limited to:
   - Tags (descriptive terms related to the product, make sure to include at least 7 tags and only use the tags available in the database, make as many tags as possible)
   - Style (e.g., casual, formal, sporty, etc.)
   - Category (e.g., Shirts, Jackets, Shoes, Sweaters, Dresses, Pants, Skirts, Shorts, Bags, Accessories, etc.)
   - Colours (detailed colors information with family, specific name, Hex code, percentage of that colour from the product) Note that the product may have multiple colours so it is array of objects, make sure that the total of all percentages is 100.
   - Material (e.g., cotton, leather, polyester, etc.)
   - A well-formatted description based on the product's attributes

2. Match the extracted attributes against the provided database: {csv_data}. Use only the data available in the database for identifying styles, categories, materials, and other attributes.

3. Provide the output in the following JSON format:
   {{
     "tags": ["tag1", "tag2", ...],
     "style": "Identified style from the database", No null values
     "category": "Identified category from the database, be as specific as possible and avoid general categories and very accurate", no null values
     "colours": [
       {{"family": "color_family", "colourName": "specific_name", "Hex": "#hex_code", "percentage": "eg. 70"}},
       ...
     ],
     "material": "Identified material from the database", no null values
     "description": "A concise and well-structured description based on the identified features, attributes, and database information, make sure to include all the attributes in the description."
   }}

4. If the image cannot be matched exactly to a product in the database, provide attributes based on the closest matches available (e.g., similar style, category, or colors).

5. Ensure all analyses and outputs strictly adhere to the attributes available in the provided database, again only from the provided database.

6. Analyze the image and provide the output in a timely manner for only one product, if you recognize more than one product it is not a problem analyse the main one in the image (the focused one), ensuring that the response is accurate and relevant to the input image.

7. Be as detailed and accurate as possible in your analysis and description, ensuring that there is no attribute in your response that is not present in the provided database or null.
"""

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

# Generate the analysis using the configured model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=system_instruction,
)

async def analyze_image_controller(file):
    """
    Controller function to upload the image to Gemini and perform analysis.
    """
    if not file.content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image.")

    temp_file_path = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Upload the file to Gemini
        uploaded_file = genai.upload_file(temp_file_path, mime_type=file.content_type)
        print(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

        # Start a chat session with the uploaded image
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                }
            ]
        )

        # Send a message to analyze the uploaded image
        response = chat_session.send_message("Analyze this image.")

        # Parse the raw response text
        try:
            input_data = json.loads(response.text)  # Deserialize the raw JSON response
        except json.JSONDecodeError:
            raise ValueError("Failed to parse the response text as JSON.")

        # Ensure the "analysis" field is properly deserialized if needed
        if isinstance(input_data.get("analysis"), str):
            input_data["analysis"] = json.loads(input_data["analysis"])

        # Pretty-print the result for readability
        formatted_json = json.dumps(input_data, indent=4)

        return formatted_json

    except Exception as e:
        raise ValueError(f"An error occurred during image analysis: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
