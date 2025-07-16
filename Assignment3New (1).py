# Databricks notebook source
# MAGIC %pip uninstall -y spacy numpy
# MAGIC %pip install --upgrade pip setuptools wheel
# MAGIC
# MAGIC # Use versions compatible with spaCy 3.7
# MAGIC %pip install numpy==1.26.4
# MAGIC %pip install spacy==3.7.2
# MAGIC !python -m spacy download en_core_web_sm
# MAGIC

# COMMAND ----------

# MAGIC %pip install transformers torch torchvision sentencepiece pillow
# MAGIC

# COMMAND ----------

import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
import requests
from PIL import Image
from io import BytesIO

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Sentiment classifier
sentiment_pipeline = pipeline("sentiment-analysis")

# 2. NER pipeline (HuggingFace)
ner_pipeline = hf_pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 3. Translation pipeline: English → French
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# 4. Image captioning model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Classify intent
def classify_intent(text):
    text = text.lower()
    if any(x in text for x in ["show me an image", "generate image", "image of", "picture of"]):
        return "image"
    elif any(x in text for x in ["translate", "french", "convert to"]):
        return "translation"
    else:
        return "text"

# Translate to French
def translate_to_french(text):
    tokens = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translation_model.generate(**tokens)
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)

# Generate image and captions
def generate_image_and_captions():
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/640px-Sunflower_sky_backdrop.jpg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)

    captions = []
    for _ in range(3):
        out = blip_model.generate(**inputs, num_beams=5, max_length=30, do_sample=True, top_p=0.9)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

    #image.show()  # May pop up locally or not work in all Databricks runtimes
    display(image)
    return captions

# Chatbot loop
print("🤖 Chatbot is ready. Type 'exit' to stop.")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("🤖 Chatbot: Goodbye!")
        break

    # Sentiment
    sentiment = sentiment_pipeline(user_input)[0]["label"]

    # NER
    ner_results = ner_pipeline(user_input)
    named_entities = [(ent["word"], ent["entity_group"]) for ent in ner_results]

    # Intent
    intent = classify_intent(user_input)

    print(f"\n[🔍 Analysis]")
    print(f"Sentiment: {sentiment}")
    print(f"Named Entities: {named_entities}")
    print(f"Intent: {intent}")

    print("\n[🤖 Response]")
    if intent == "text":
        print("Here's a textual response to your prompt.")
    elif intent == "image":
        captions = generate_image_and_captions()
        for i, cap in enumerate(captions, 1):
            print(f"Caption {i}: {cap}")
    elif intent == "translation":
        translated = translate_to_french(user_input)
        print(f"French Translation: {translated}")
    else:
        print("Sorry, I couldn't understand your request.")
    
    print("\n" + "="*60 + "\n")
