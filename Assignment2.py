# Databricks notebook source
pip install diffusers


# COMMAND ----------

# %pip install transformers sentence-transformers accelerate

# COMMAND ----------

# MAGIC %pip install diffusers[torch] transformers accelerate safetensors --upgrade
# MAGIC

# COMMAND ----------

# Imports
from transformers import pipeline, set_seed
from diffusers import StableDiffusionPipeline
import torch

# Force CPU use
device = "cpu"

# Set seed for reproducibility
set_seed(42)

# -------------------------
# 🔹 PL1: Text Enhancement
# -------------------------
text_enhancer = pipeline("text-generation", model="gpt2", device=-1)

user_prompt = "A quiet lake under the moonlight"
print(f"🔹 Prompt (PL1 input): {user_prompt}")

enhanced_text = text_enhancer(user_prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
print(f"✅ PL1 Output (Enhanced Text):\n{enhanced_text}\n")

# -------------------------
# 🔹 PL2: Extract Key-Phrases
# -------------------------
# Using summarization to get short keywords (simplified)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
summary = summarizer(enhanced_text, max_length=20, min_length=5, do_sample=False)[0]["summary_text"]

print(f"✅ PL2 Output (Key Phrases):\n{summary}\n")

# -------------------------
# 🔹 PL3: Create a new story or poem
# -------------------------
story_gen = pipeline("text-generation", model="gpt2", device=-1)

prompt_for_poem = f"Write a poetic line based on: {summary}"
new_story = story_gen(prompt_for_poem, max_length=50, num_return_sequences=1)[0]["generated_text"]

print(f"✅ PL3 Output (Story/Poem):\n{new_story}\n")

# -------------------------
# 🔹 PL4: Generate Image (CPU version)
# -------------------------
print("🧠 Generating image based on PL3 output (this may take a few minutes on CPU)...")

image_prompt = new_story

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cpu")  # Ensure model is on CPU

image = pipe(image_prompt, num_inference_steps=25).images[0]

# Save & display
image.save("generated_image.png")
display(image)
