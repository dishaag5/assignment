# Databricks notebook source
# MAGIC %md
# MAGIC # 1st way

# COMMAND ----------

#1st way 
pip install PyMuPDF numpy


# COMMAND ----------

import fitz  # PyMuPDF
import numpy as np
import os


# Load PDF and extract text
def extract_words_from_pdf(pdf_path, max_pages=2):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(min(max_pages, len(doc))):
        text += doc[page_num].get_text()
    words = [word.strip(".,!?;:\"'()[]") for word in text.split()]
    return [word for word in words if word]

# Create dummy embeddings
def embed_word(word, dim, seed=42):
    rng = np.random.default_rng(hash(word + str(seed)) % (2**32))
    return rng.normal(size=dim).astype(np.float32)

# Main pipeline
def process_pdf_to_embeddings(pdf_path):
    words = extract_words_from_pdf(pdf_path)

    files = {
        8: open("embeddings_8d.txt", "w"),
        16: open("embeddings_16d.txt", "w"),
        32: open("embeddings_32d.txt", "w")
    }

    for word in words:
        for dim in files:
            embedding = embed_word(word, dim)
            files[dim].write(f"{word}\t{','.join(map(str, embedding))}\n")

    for f in files.values():
        f.close()

# Call the function
process_pdf_to_embeddings("/Workspace/Users/dbuser14@meteoros.ai/XGBoost_WM.pdf")  # Replace with your PDF path


# COMMAND ----------

# MAGIC %md
# MAGIC ##  2nd way

# COMMAND ----------

## 2nd way:
!pip install sentence-transformers PyMuPDF scikit-learn

    

# COMMAND ----------

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

# === Step 1: Load a PDF and extract text ===
def read_pdf_words(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    words = list(set(full_text.split()))  # Unique words only
    return words

# === Step 2: Generate embeddings using BERT ===
def generate_bert_embeddings(words):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Output dim: 384
    embeddings = model.encode(words, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# === Step 3: Reduce embeddings to target dims (8, 16, 32) ===
def reduce_embeddings(embeddings, target_dim):
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(embeddings)
    return reduced

# === Step 4: Write to file ===
def write_to_file(words, embeddings, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for word, emb in zip(words, embeddings):
            emb_str = " ".join(f"{v:.4f}" for v in emb)
            f.write(f"{word}\t{emb_str}\n")

# === Main pipeline ===
def main(pdf_path):
    words = read_pdf_words(pdf_path)
    print(f"Total unique words: {len(words)}")

    original_embeddings = generate_bert_embeddings(words)

    for dim in [8, 16, 32]:
        reduced = reduce_embeddings(original_embeddings, dim)
        write_to_file(words, reduced, f"embeddingss_{dim}d.txt")
        print(f"Saved {dim}-D embeddings to embeddingss_{dim}d.txt")

# === Run it ===
if __name__ == "__main__":
    main("/Workspace/Users/dbuser14@meteoros.ai/XGBoost_WM.pdf")  # <-- Replace with your actual PDF path


# COMMAND ----------

# MAGIC %md
# MAGIC 3rd way full code

# COMMAND ----------

import fitz  # PyMuPDF for PDF reading
import numpy as np
import os

# ---------- 1. Read and extract unique words from PDF ----------
def extract_words_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    words = text.split()
    clean_words = [w.strip('.,!?()[]{}:;"\'').lower() for w in words if w.isalpha()]
    return list(set(clean_words))  # return unique words

# ---------- 2. Generate fixed random embedding using word hash ----------
def generate_embedding(word, dim):
    np.random.seed(abs(hash(word)) % (10 ** 8))  # hash-based consistent seed
    return np.random.rand(dim)

# ---------- 3. Save all words and their embeddings into 3 files ----------
def save_embeddings(words, dims, paths):
    for dim, path in zip(dims, paths):
        with open(path, 'w') as f:
            for word in words:
                emb = generate_embedding(word, dim)
                emb_str = ' '.join(map(str, emb))
                f.write(f"{word} {emb_str}\n")

# ---------- 4. Load existing embeddings from a file ----------
def load_embeddings(path):
    emb_dict = {}
    if not os.path.exists(path):
        return emb_dict
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            emb_dict[word] = np.array(vector)
    return emb_dict

# ---------- 5. Search or generate embedding for a given word ----------
def search_or_generate(word, dims, paths):
    found = False
    results = []

    for dim, path in zip(dims, paths):
        emb_dict = load_embeddings(path)

        if word in emb_dict:
            print(f"[{dim}D] Found: {emb_dict[word]}")
            results.append(emb_dict[word])
            found = True
        else:
            print(f"[{dim}D] '{word}' not found, generating...")
            new_emb = generate_embedding(word, dim)
            results.append(new_emb)
            with open(path, 'a') as f:
                emb_str = ' '.join(map(str, new_emb))
                f.write(f"{word} {emb_str}\n")

    if not found:
        print("Word was not previously embedded. New embeddings generated and stored.")

    return results

# ---------- 6. MAIN EXECUTION ----------

# Config
pdf_file = '/Workspace/Users/dbuser14@meteoros.ai/XGBoost_WM.pdf'  # replace with the path to your PDF
embedding_dims = [8, 16, 32]
embedding_files = ['embed_8d.txt', 'embed_16d.txt', 'embed_32d.txt']

# Step 1: Extract and embed all unique words from PDF
pdf_words = extract_words_from_pdf(pdf_file)
save_embeddings(pdf_words, embedding_dims, embedding_files)
print("✅ Initial embeddings generated from PDF.")

# Step 2: Accept input and search/generate embeddings
input_word = input("Enter a word to search for its embedding: ").strip().lower()
search_or_generate(input_word, embedding_dims, embedding_files)
