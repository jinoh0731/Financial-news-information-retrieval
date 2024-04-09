# Financial-news-information-retrieval

# Introduction

## Purpose of the Project

## What is Information Retrieval (IR)?

Information Retrieval (IR) is the discipline of searching for information resources relevant to an information need from a large collection of information resources. This typically involves retrieving documents or other content from a database, web, or any organized repository. The main goal of IR is to provide easy, fast, and effective access to large volumes of information.

### Key Components of IR

- **Queries**: A query is a formal statement of information needs, such as a search string that users input into a search system.
- **Documents**: In IR, documents can refer to any piece of textual information. They can be web pages, news articles, scientific papers, books, emails, or any other kind of textual data.
- **Indexing**: Indexing is the process of creating a map between user queries and the documents stored in the database. It involves parsing the documents and building a searchable data structure, typically an index, from the terms found in the documents.
- **Search Algorithm**: The search algorithm takes a user query and returns a ranked list of documents based on their relevance to the query. This ranking is often based on metrics like frequency of query terms in documents, document length, and other factors.
- **Relevance**: Relevance measures how well a document corresponds to a query's intent. It is a critical aspect in the effectiveness of an IR system.


  
# Dataset

GPT4 for labels

# Models
## 1. ADA
## 2. Two Towers
### 2-1 BERT
### 2-2 RoBERTa

## 3. MiniLM (all-MiniLM-L6-v2)
The `all-MiniLM-L6-v2` is a model from the Sentence Transformers library, which is designed to produce state-of-the-art sentence embeddings. This model is a smaller, faster version of BERT and is optimized for generating embeddings that are particularly useful for tasks involving semantic similarity.

- **MiniLM Architecture**: MiniLM stands for Miniature Language Model. It focuses on reducing the model size while retaining a high level of performance. The model is designed by Microsoft and leverages knowledge distillation techniques where a smaller model is trained to reproduce the behavior of a much larger one.

- **Training Objective and Application:**

  - **BERT**: Trained on a large corpus with objectives like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). It is primarily designed for improving performance on a variety of NLP tasks such as question answering and natural language inference by fine-tuning.
  - **MiniLM**: While also trained with MLM, MiniLM models are particularly fine-tuned for producing sentence embeddings. They are optimized through knowledge distillation from a teacher model (usually a larger BERT or RoBERTa model) and are specifically fine-tuned to improve performance in tasks where sentence-level embeddings are used directly, like semantic textual similarity.


- **Layer Count (L6)**: The "L6" in the model name indicates that it utilizes 6 transformer layers, which is fewer than the larger models like BERT-base or RoBERTa which typically use 12 layers. This reduction in layers significantly decreases the computational resources needed without a substantial drop in performance.

- **Efficiency**: Due to its reduced size, all-MiniLM-L6-v2 is more efficient in terms of memory usage and speed. This makes it ideal for environments with limited computational resources or applications that require real-time processing.

- **Performance**: Despite its smaller size, the model continues to deliver strong performance across a variety of NLP tasks, particularly those involving semantic understanding of text such as similarity measurement, clustering, and information retrieval.



# Model Performance

# Code Demo

# Gradio Demo

# Limitation & Future Word

# Resource
