# Databricks notebook source
# MAGIC %md
# MAGIC # 1st way

# COMMAND ----------

#1st way 
pip install PyMuPDF numpy


# COMMAND ----------

import fitz  # PyMuPDF
import numpy as np

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
