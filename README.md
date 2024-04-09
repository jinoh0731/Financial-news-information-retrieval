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

#### Embeddings (for my reference...)
An embedding is a sequence of numbers that represents the concepts within content such as natural language or code. Embeddings make it easy for machine learning models and other algorithms to understand the relationships between content and to perform tasks like clustering or retrieval. 

# Models
## 1. Open-AI Text Embedding (text-embedding-3-small)
![Screenshot 2024-04-08 at 8 04 17 PM](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/1f626312-cdda-44d6-87d1-584013133037)

The `text-embedding-3-small` model is the newest embedding model, released on January 25, 2024, by OpenAI to provide efficient, compact text embeddings suitable for a variety of tasks that require text representation. It is an upgrade over its predecessor, the text-embedding-ada-002 model which was released in December 2022.  This model is optimized for environments where model size and speed are critical, making it ideal for applications requiring low-latency processing.

![Screenshot 2024-04-08 at 8 08 01 PM](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/1e44a974-0faf-4ce0-8f9c-b9f79635e447)

- **Performance**:Comparing `text-embedding-ada-002` to `text-embedding-3-small`, the average score on a commonly used benchmark for multi-language retrieval (MIRACL) has increased from 31.4% to 44.0%, while the average score on a commonly used benchmark for English tasks (MTEB) has increased from 61.0% to 62.3%.

- **Compact Design**: As indicated by the name, `text-embedding-3-small` is a smaller version of larger text embedding models. It is specifically optimized to maintain a balance between performance and resource efficiency.

- **Efficient Text Embeddings**: This model generates embeddings that are smaller in size but still capture the essential semantic features needed for tasks such as text similarity, classification, and clustering.

- **Speed and Efficiency**: The model's reduced size allows for faster processing times and lower memory usage, which is beneficial for applications running on limited resources or requiring real-time response.

- **Scalability**: Due to its efficiency and compact nature, `text-embedding-3-small` can easily scale to handle large volumes of text without significant resource expenditure.

## 2. Two Towers
(introduce two towers architecture)

### 2-1 BERT (bert_en_uncased_L-12_H-768_A-12)

BERT is a groundbreaking model in the field of natural language processing (NLP) introduced by researchers at Google. BERT's key innovation is its deep bidirectionality, allowing the model to understand the context of a word based on all of its surroundings (both left and right of the word).

The `bert_en_uncased_L-12_H-768_A-12` model is a specific configuration of the BERT model developed by Google. This variant is called BERT Miniatures, and it was introduced as part of BERT's initial release in October 2018. It is designed to handle tasks that require understanding the context of English language text in a way that is agnostic to the case of the input text (i.e., lowercased input). Also, the standard BERT is effective on a wide range of model sizes, beyond BERT-Base and BERT-Large, whereas smaller BERT models are intended for environments with restricted computational resources

- **Configuration Details**:
  - **L-12**: The model consists of 12 Transformer blocks (layers), providing a good balance between computational efficiency and modeling capability.
  - **H-768**: Each layer has 768 hidden units, which contribute to the model's depth and capacity for language understanding.
  - **A-12**: The model uses 12 attention heads, allowing it to focus on different parts of the sentence simultaneously, which enhances its ability to understand complex syntactic and semantic relationships in text.

- **Uncased**: This model variant treats text in a case-insensitive manner, making it suitable for tasks where case information is not critical for understanding the content.

- **Pre-trained Knowledge**: As with all BERT models, `bert_en_uncased_L-12_H-768_A-12` benefits from pre-training on a large corpus, specifically the BooksCorpus (800M words) and English Wikipedia (2,500M words), which helps the model develop a broad understanding of the English language.


### 2-2 RoBERTa (Robustly Optimized BERT Approach)

![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/5cba2eed-751e-455f-9bcd-919a67b527cb)

The RoBERTa model was proposed in RoBERTa: A Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. RoBERTa is an optimized version of the BERT architecture, developed by Facebook AI in 2019. It builds upon BERT's methodology with changes in the pre-training procedure that significantly improve performance across a range of natural language processing (NLP) tasks. RoBERTa demonstrates that careful tuning of hyperparameters and training data size can lead to substantial improvements even within existing model architectures.

- **Modified Training Approach**: Unlike BERT, RoBERTa does not use the Next Sentence Prediction (NSP) task during training. It was found that removing NSP, and training with longer sequences/more data improved performance.
  
- **Dynamic Masking**: RoBERTa applies the masking dynamically to the training data, which means the model sees different masked versions of the same text. This contrasts with BERT, which uses a static mask once created at the data pre-processing step.

- **Larger Batch Sizes and More Data**: RoBERTa is trained with much larger batch sizes and on more data compared to BERT. For instance, it was trained on multiple datasets totaling over 160GB of uncompressed text.

- **Byte-Pair Encoding (BPE)**: RoBERTa uses a byte-level BPE as a tokenizer, which allows it to handle a wider range of characters and to be more effective in multilingual environments.


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
