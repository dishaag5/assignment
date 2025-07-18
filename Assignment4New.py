# Databricks notebook source
# MAGIC %md
# MAGIC Individual Project/Activity
# MAGIC In this project, we’re going to step into territory we haven’t covered before: LangChain Runnables. You’ll use CodeAssist to help you work with this new abstraction, which powers modern, composable LLM workflows. Your task is to build a simple RAG pipeline that can dynamically switch between semantic retrieval and function calling—based entirely on what kind of prompt the user provides.
# MAGIC Start by collecting three PDF files on any topic you enjoy—climate change, computer architecture, economics, mythology—anything you’d like to ask questions about. Extract raw text from each PDF and chunk it into smaller passages of around 300–500 characters. Store these chunks in a Hive Metastore-backed table using Spark. Next, generate vector embeddings for each chunk using OpenAI’s text-embedding-ada-002 model, and store them either in the same Hive table or in a new one. This gives you a searchable document index backed by Hive.
# MAGIC Now it’s time to build your retrieval logic—but instead of using standard Python functions, you’ll build it using LangChain Runnables. You’re expected to define a runnable that takes an input question, embeds it, compares it with your stored vectors, and returns the top matching documents. Use CodeAssist to help you write and organize your Runnables properly—especially if you’re not sure how to structure them or chain them together. This part will likely include combining components like embedding models, retrievers, and prompt templates into a runnable pipeline.
# MAGIC Here’s the twist: your system must be able to recognize if the user is asking a search question or a math question. If the user types a prompt like “What are the causes of inflation?” then your RAG pipeline should run as usual. But if the user types “What is 234 + 91?”, you should skip retrieval and instead call a custom Python function called addition(a, b) that returns the sum. You’ll use a simple prompt classification step to make this decision—either using keyword rules or a lightweight GPT-4 prompt that returns a task type. Code generation using GPT-4 can help you build this logic if you’re unsure how to write clean and testable decision code.
# MAGIC By the end of the project, you’ll have built a hybrid LLM pipeline that can switch between knowledge-grounded generation and direct computation, powered by LangChain and deployed entirely in Databricks. You’ll use CodeAssist to support you in writing unfamiliar LangChain code, GPT-4 to help generate or correct logic, and the Hive Metastore to manage your vectorized knowledge base. The outcome is a flexible, AI-assisted system that demonstrates not just retrieval, but smart reasoning about when retrieval is appropriate.
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC Everyone
# MAGIC  
# MAGIC Azure OpenAI Endpoint:https://kiran255666.openai.azure.com/
# MAGIC  
# MAGIC Azure openAI Key: 9qwAgDEDJJPsCidbF5tV6UJoK3y33t5jqpt64N1SGOjrXtByiVSJJQQJ99BGAC4f1cMXJ3w3AAABACOGIox2
# MAGIC  
# MAGIC GPT model name: gpt-4o
# MAGIC  
# MAGIC Text embedding model name:  text-embedding-ada-002
# MAGIC  

# COMMAND ----------

# Install dependencies
%pip install pypdf langchain openai scikit-learn


# COMMAND ----------

from openai import AzureOpenAI

# 🔹 Set your Azure OpenAI credentials
AZURE_ENDPOINT = "https://kiran255666.openai.azure.com/"
AZURE_API_KEY = "9qwAgDEDJJPsCidbF5tV6UJoK3y33t5jqpt64N1SGOjrXtByiVSJJQQJ99BGAC4f1cMXJ3w3AAABACOGIox2"
AZURE_API_VERSION = "2024-12-01-preview"

# 🔹 Your deployed model names (NOT base models!)
AZURE_CHAT_DEPLOYMENT = "gpt-4o"              # chat completion
AZURE_EMBED_DEPLOYMENT = "text-embedding-ada-002"  # embeddings

# Azure client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# Test chat
resp = client.chat.completions.create(
    model=AZURE_CHAT_DEPLOYMENT,
    messages=[{"role": "user", "content": "Hello Azure!"}]
)
print(resp.choices[0].message.content)


# COMMAND ----------

from pypdf import PdfReader
import re

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def chunk_text(text, chunk_size=400):
    text = re.sub(r"\s+", " ", text)  # clean whitespace
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Example PDFs
pdf_files = ["x.pdf",
             "y.pdf",
             "z.pdf"]

all_chunks = []
for f in pdf_files:
    raw_text = extract_text_from_pdf(f)
    chunks = chunk_text(raw_text)
    all_chunks.extend([(f.split("/")[-1], i, chunk) for i, chunk in enumerate(chunks)])  # (doc, id, text)

print(f"✅ Total chunks: {len(all_chunks)}")


# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

df_chunks = spark.createDataFrame(all_chunks, ["doc_name", "chunk_id", "content"])
df_chunks.write.mode("overwrite").saveAsTable("rag_pdf_chunks_og")


# COMMAND ----------

def embed_text(text: str):
    response = client.embeddings.create(
        model=AZURE_EMBED_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

# Generate embeddings
embedded_chunks = [
    (doc, cid, text, embed_text(text))
    for doc, cid, text in all_chunks
]

# Convert to Spark-friendly (array<float>)
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField

schema = StructType([
    StructField("doc_name", StringType(), True),
    StructField("chunk_id", StringType(), True),
    StructField("content", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True)
])

df_embed = spark.createDataFrame(
    [(doc, str(cid), text, [float(x) for x in vec]) for doc, cid, text, vec in embedded_chunks],
    schema
)

df_embed.write.mode("overwrite").saveAsTable("rag_pdf_chunks_embedded_og")


# COMMAND ----------

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pdf_data = df_embed.collect()
pdf_chunks = [(row.content, np.array(row.embedding)) for row in pdf_data]

def retrieve_top_chunks(query, k=3):
    q_emb = np.array(embed_text(query))
    sims = [(text, cosine_similarity([q_emb], [emb])[0][0]) for text, emb in pdf_chunks]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return [s[0] for s in sims[:k]]


# COMMAND ----------

def addition(a, b):
    return a + b

def handle_math(prompt: str):
    nums = [int(x) for x in prompt.split() if x.isdigit()]
    if len(nums) >= 2:
        return f"The sum is {addition(nums[0], nums[1])}"
    return "I can only handle simple two-number addition right now."


# COMMAND ----------

def classify_prompt(prompt: str) -> str:
    if any(op in prompt for op in ["+", "-", "*", "/"]) and any(c.isdigit() for c in prompt):
        return "math"
    return "search"


# COMMAND ----------

def generate_answer_with_context(question: str, docs: list):
    context = "\n\n".join(docs)
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


# COMMAND ----------

# from langchain.schema.runnable import RunnableLambda, RunnableBranch

# # 1️⃣ Wrap everything as Runnables
# classifier_runnable = RunnableLambda(classify_prompt)
# retriever_runnable = RunnableLambda(lambda q: retrieve_top_chunks(q))
# math_runnable = RunnableLambda(handle_math)

# answer_runnable = RunnableLambda(lambda inp: generate_answer_with_context(inp["query"], inp["docs"]))

# # Combine retriever + GPT answer
# retrieval_chain = retriever_runnable | RunnableLambda(
#     lambda docs: {"query": docs.get("prompt", "unknown query"), "docs": docs if isinstance(docs, list) else [docs]}
# ) | answer_runnable

# # ✅ Proper RunnableBranch
# branch_runnable = RunnableBranch(
#     # Branch 1 → Math
#     (
#         lambda prompt: classify_prompt(prompt) == "math",
#         math_runnable
#     ),
#     # Branch 2 → Retrieval RAG
#     (
#         lambda prompt: classify_prompt(prompt) == "search",
#         retrieval_chain
#     ),
#     # ✅ Default → always runs if no other matches
#     RunnableLambda(lambda x: "Sorry, I don't understand this request.")
# )


# COMMAND ----------

from langchain.schema.runnable import RunnableLambda, RunnableBranch

# Math branch
math_runnable = RunnableLambda(handle_math)

# Retrieval now returns dict {"query": prompt, "docs": [chunks]}
retriever_runnable = RunnableLambda(
    lambda prompt: {"query": prompt, "docs": retrieve_top_chunks(prompt)}
)

# Final GPT call → returns a string answer
answer_runnable = RunnableLambda(
    lambda inp: generate_answer_with_context(inp["query"], inp["docs"])
)

# Retrieval + GPT chain
retrieval_chain = retriever_runnable | answer_runnable

# Classifier runnable (optional, just for debugging)
classifier_runnable = RunnableLambda(classify_prompt)

# Branch logic
branch_runnable = RunnableBranch(
    # 1️⃣ Math question → directly call math_runnable
    (
        lambda prompt: classify_prompt(prompt) == "math",
        math_runnable
    ),
    # 2️⃣ Knowledge question → retrieval chain
    (
        lambda prompt: classify_prompt(prompt) == "search",
        retrieval_chain
    ),
    # ✅ Default → always runs if no match
    RunnableLambda(lambda x: "Sorry, I don't understand this request.")
)


# COMMAND ----------

# query = "What are the main causes of inflation?"

# # Call ONCE and store result
# result = branch_runnable.invoke(query)

# # If retrieval branch returns a list of docs, just pick first; else print as-is
# if isinstance(result, list):
#     response = result[0]
# else:
#     response = result

# print("Response:", response)

# # Test math branch
# query2 = "What is 234 + 91?"
# result2 = branch_runnable.invoke(query2)
# print("Response2:", result2)


# COMMAND ----------

query1 = "What are the main causes of inflation?"
response1 = branch_runnable.invoke(query1)
print("RAG Answer:", response1)

query2 = "What is 234 + 91?"
response2 = branch_runnable.invoke(query2)
print("Math Answer:", response2)

query3 = "Just saying hi!"
response3 = branch_runnable.invoke(query3)
print("Default Answer:", response3)


# COMMAND ----------

query4 = "What are strings in python?"
response4 = branch_runnable.invoke(query4)
print("RAG Answer:", response4)
