# Databricks notebook source
# MAGIC %md
# MAGIC Build a chatbot (infinite while loop with an exit command) which can do the following:
# MAGIC 1. Classify input prompt as a positive or negative sentiment
# MAGIC 2. Extract all Named Entities from the prompt
# MAGIC 3. Classify if the prompt is requesting text information or image or translation
# MAGIC a. Appropriately return image or text as response
# MAGIC b. If the request is for an image, the bot should return at least 3 captions along
# MAGIC with the image
# MAGIC c. Translate from English to any other language of your choice
# MAGIC

# COMMAND ----------

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

!pip install diffusers transformers accelerate torch safetensors

# COMMAND ----------

import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from transformers import pipeline as hf_pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from IPython.display import display

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

# 5. SMALL diffusion model for image generation
print("⏳ Loading lightweight diffusion model...")
image_gen_pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",  # Much faster & smaller than full SD
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
print("✅ Image generation model ready!")

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

# Generate image dynamically from text
def generate_image_from_text(prompt):
    print(f"🎨 Generating an image for: {prompt}")
    image = image_gen_pipeline(prompt, num_inference_steps=15).images[0]
    image.save("generated_image.jpg")
    return image

# Generate captions for an image
def generate_captions(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    captions = []
    for _ in range(2):
        out = blip_model.generate(**inputs, num_beams=3, max_length=30, do_sample=True, top_p=0.9)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
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
        # Generate an image dynamically
        image_prompt = user_input.replace("show me", "").replace("generate image of", "").strip()
        image = generate_image_from_text(image_prompt)
        #image.show()  # Opens locally (works in notebooks/desktop)
        display(image)
        
        # Now generate captions
        captions = generate_captions(image)
        for i, cap in enumerate(captions, 1):
            print(f"Caption {i}: {cap}")
    elif intent == "translation":
        translated = translate_to_french(user_input)
        print(f"French Translation: {translated}")
    else:
        print("Sorry, I couldn't understand your request.")
    
    print("\n" + "="*60 + "\n")
