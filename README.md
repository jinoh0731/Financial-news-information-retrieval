# Financial-news-information-retrieval

# Introduction

![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/123cded9-65f1-4e9c-b0db-3c55bc05dc7d)


## Problem Statement
ESG (Environmental Social Governance) investing has been gaining an increasing interest over the years as more people wish to contribute positively to the environment while mitigating portfolio risks and generating long-term portfolio value. With a growing interest in ESG investing, asset managers need to rapidly analyze news related to ESG topics to gain timely insights and make informed decisions. However, there are unclear standards in what is ESG-relevant news as the unstructured nature of news articles makes quickly extracting relevant information difficult. Given that the stock market reflects news very quickly, the ability to efficiently and accurately extract relevant ESG news not only becomes a critical competitive advantage, but it can also help asset managers construct better ESG focused portfolios.

## Approach
To address above challenges, this project employs advanced information retrieval (IR) models designed to efficiently sift through and accurately identify ESG-related news content. This approach could help asset managers quickly retrieve ESG-related news for a given industry to quickly get actionable insights when making investment decisions. We have selected four different information retrieval (IR) models and determine the best IR model that can help asset managers get relevant ESG news for each industry in a simple manner.

1. [Two Towers](#1-two-towers)
   - [1.1 BERT (bert_en_uncased_L-12_H-768_A-12)](#1-1-bert-bert_en_uncased_l-12_h-768_a-12)
   - [1.2 RoBERTa (Robustly Optimized BERT Approach)](#1-2-roberta-robustly-optimized-bert-approach)
2. [MiniLM (all-MiniLM-L6-v2)](#2-minilm-all-minilm-l6-v2)
3. [Open-AI Text Embedding (text-embedding-3-small)](#3-open-ai-text-embedding-text-embedding-3-small)



## What is Information Retrieval (IR)?

Information Retrieval (IR) is the discipline of searching for information resources relevant to an information need from a large collection of information resources. This typically involves retrieving documents or other content from a database, web, or any organized repository. The main goal of IR is to provide easy, fast, and effective access to large volumes of information.

For this project, we are conducting IR to extract relevant ESG news data to reduce the noise found in large news datasets. This will help asset managers get news articles that are most ESG relevant, which will help asset managers make informed investment decisions in portfolio construction. 

### Key Components of IR

- **Queries**: A query is a formal statement of information needs, such as a search string that users input into a search system.
- **Documents**: In IR, documents can refer to any piece of textual information. They can be web pages, news articles, scientific papers, books, emails, or any other kind of textual data.
- **Indexing**: Indexing is the process of creating a map between user queries and the documents stored in the database. It involves parsing the documents and building a searchable data structure, typically an index, from the terms found in the documents.
- **Search Algorithm**: The search algorithm takes a user query and returns a ranked list of documents based on their relevance to the query. This ranking is often based on metrics like frequency of query terms in documents, document length, and other factors.
- **Relevance**: Relevance measures how well a document corresponds to a query's intent. It is a critical aspect in the effectiveness of an IR system.


# Dataset (Incomplete)

The dataset is comprised of three different parts:
- **Perigon**: Perigon News Data is part of the broader set of services offered by Perigon, specifically focusing on delivering real-time, AI-enriched global news data. This service provides access to a vast and diverse range of news sources, encompassing over 100,000 sources worldwide. Dataset has 5,216 news articles with a range from 9/13/2022 to 9/8/2023.
   - Relevant Columns:
     - `title_and_content` – News article
     - `Industry` – A categorical variable to represent the news article’s main industry. There are 67 categories.
     - `url` – URL of the news article
     - `articleId` – unique identifier within the Perigon News Data

- **Sustainability Accounting Standards Board (SASB)**: SASB is a non-profit organization that develops and maintains industry-specific standards that guide companies' disclosure of financially material sustainability information to investors and other financial stakeholders. Data was webscraped from 67 industries that identified the ESG issues relevant to a particular industry. For each industry, there are 3-8 sub-topics that indicate what is considered ESG within that particular industry.
- **GPT4 for labels**: To identify whether an article is ESG relevant or not, we needed to create ground truth labels for our dataset. We created a GPT4 pipeline using the following prompts:
o	INSERT PROMPT
GPT4 would then label an article as “Major”, “Minor” or “No” ESG relevant, which we used as ground truth for the rest of our dataset.


During data pre-processing, we applied Spacy to remove any stop-words to end up with the following word count:
Graphic of Cleaned News Articles Word Count:
![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295407/8b350c42-0a95-4301-8ebc-ea342e9cd199)

Graphic of Cleaned SASB Query Word Count:
![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295407/daa2be20-efa3-4e4d-84a5-88a3b4cce0e8)

##### Side notes for removal later:
Notice that for certain situations, we do extend past the context length. (Should we make a chart on the maximum context length sizes? Alternatively, there is the new suggestion of reducing context with ChatGPT)

#### Embeddings (for my reference...)
An embedding is a sequence of numbers that represents the concepts within content such as natural language or code. Embeddings make it easy for machine learning models and other algorithms to understand the relationships between content and to perform tasks like clustering or retrieval. 

# Models

## 1. Two Towers
(introduce two towers architecture - may adjust later if it doesn't make sense)

Two Towers is a dual encoder or bi-encoder architecture that employs two separate "towers" or neural networks to learn embeddings for queries and documents independently. The architecture allows for the nuanced capturing of semantic relationships between candiates and queries that is seen when we compute the cosine similarity score between each candidate and query pair.
   - Key Components:
      - Candidate: The candidate is the news article that excludes stop words.
      - Query: The query is the concatenation of all the SASB topics for a specific industry after excluding stop words. We chose this query formation methodology to have the best approximation of what is considered ESG for a given industry, regardless if it is "Major" or "Minor" ESG related. There are a total of 67 queries to represent the 67 total possible industries in our dataset.
      - Model will return a cosine similarity score that represents the model's predictions on what articles are considered ESG-relevant for a given query. We can use the model's output cosine similarity score to rank articles based on their similarity to a given query.

- Benefits:
   - With candidate and query each separatedly sent to different towers, we can increase the total amount of words sent to the encoder.
   - Utilized encoder models with Two Towers are free-to-use.
   - Architecture and encoder models are able to get fine-tune - DOUBLE CHECK FOR roBERTa

Two Towers Example Diagram:
![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295407/469541f4-c9ed-4d35-9c2d-d3d6ecf7adbb)

### 1-1 BERT (bert_en_uncased_L-12_H-768_A-12)

BERT is a groundbreaking model in the field of natural language processing (NLP) introduced by researchers at Google. BERT's key innovation is its deep bidirectionality, allowing the model to understand the context of a word based on all of its surroundings (both left and right of the word).

The `bert_en_uncased_L-12_H-768_A-12` model is a specific configuration of the BERT model developed by Google. This variant is called BERT Miniatures, and it was introduced as part of BERT's initial release in October 2018. It is designed to handle tasks that require understanding the context of English language text in a way that is agnostic to the case of the input text (i.e., lowercased input). Also, the standard BERT is effective on a wide range of model sizes, beyond BERT-Base and BERT-Large, whereas smaller BERT models are intended for environments with restricted computational resources

- **Configuration Details**:
  - **L-12**: The model consists of 12 Transformer blocks (layers), providing a good balance between computational efficiency and modeling capability.
  - **H-768**: Each layer has 768 hidden units, which contribute to the model's depth and capacity for language understanding.
  - **A-12**: The model uses 12 attention heads, allowing it to focus on different parts of the sentence simultaneously, which enhances its ability to understand complex syntactic and semantic relationships in text.

- **Uncased**: This model variant treats text in a case-insensitive manner, making it suitable for tasks where case information is not critical for understanding the content.

- **Pre-trained Knowledge**: As with all BERT models, `bert_en_uncased_L-12_H-768_A-12` benefits from pre-training on a large corpus, specifically the BooksCorpus (800M words) and English Wikipedia (2,500M words), which helps the model develop a broad understanding of the English language.


### 1-2 RoBERTa (Robustly Optimized BERT Approach)

![image](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/5cba2eed-751e-455f-9bcd-919a67b527cb)

The `RoBERTa` model was proposed in *RoBERTa: A Robustly Optimized BERT Pretraining Approach* by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. `RoBERTa` is an optimized version of the BERT architecture, developed by Facebook AI in 2019. It builds upon BERT's methodology with changes in the pre-training procedure that significantly improve performance across a range of natural language processing (NLP) tasks. RoBERTa demonstrates that careful tuning of hyperparameters and training data size can lead to substantial improvements even within existing model architectures.

- **Modified Training Approach**: Unlike BERT, RoBERTa does not use the Next Sentence Prediction (NSP) task during training. It was found that removing NSP, and training with longer sequences/more data improved performance.
  
- **Dynamic Masking**: RoBERTa applies the masking dynamically to the training data, which means the model sees different masked versions of the same text. This contrasts with BERT, which uses a static mask once created at the data pre-processing step.

- **Larger Batch Sizes and More Data**: RoBERTa is trained with much larger batch sizes and on more data compared to BERT. For instance, it was trained on multiple datasets totaling over 160GB of uncompressed text.

- **Byte-Pair Encoding (BPE)**: RoBERTa uses a byte-level BPE as a tokenizer, which allows it to handle a wider range of characters and to be more effective in multilingual environments.


## 2. MiniLM (all-MiniLM-L6-v2)
The `all-MiniLM-L6-v2` is a model from the Sentence Transformers library, which is designed to produce state-of-the-art sentence embeddings. This model is a smaller, faster version of BERT and is optimized for generating embeddings that are particularly useful for tasks involving semantic similarity.

- **MiniLM Architecture**: MiniLM stands for Miniature Language Model. It focuses on reducing the model size while retaining a high level of performance. The model is designed by Microsoft and leverages knowledge distillation techniques where a smaller model is trained to reproduce the behavior of a much larger one.

- **Training Objective and Application:**

  - **BERT**: Trained on a large corpus with objectives like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). It is primarily designed for improving performance on a variety of NLP tasks such as question answering and natural language inference by fine-tuning.
  - **MiniLM**: While also trained with MLM, MiniLM models are particularly fine-tuned for producing sentence embeddings. They are optimized through knowledge distillation from a teacher model (usually a larger BERT or RoBERTa model) and are specifically fine-tuned to improve performance in tasks where sentence-level embeddings are used directly, like semantic textual similarity.


- **Layer Count (L6)**: The "L6" in the model name indicates that it utilizes 6 transformer layers, which is fewer than the larger models like BERT-base or RoBERTa which typically use 12 layers. This reduction in layers significantly decreases the computational resources needed without a substantial drop in performance.

- **Efficiency**: Due to its reduced size, all-MiniLM-L6-v2 is more efficient in terms of memory usage and speed. This makes it ideal for environments with limited computational resources or applications that require real-time processing.

- **Performance**: Despite its smaller size, the model continues to deliver strong performance across a variety of NLP tasks, particularly those involving semantic understanding of text such as similarity measurement, clustering, and information retrieval.


## 3. Open-AI Text Embedding (text-embedding-3-small)
![Screenshot 2024-04-08 at 8 04 17 PM](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/1f626312-cdda-44d6-87d1-584013133037)

The `text-embedding-3-small` model is the newest embedding model, released on January 25, 2024, by OpenAI to provide efficient, compact text embeddings suitable for a variety of tasks that require text representation. It is an upgrade over its predecessor, the text-embedding-ada-002 model which was released in December 2022.  This model is optimized for environments where model size and speed are critical, making it ideal for applications requiring low-latency processing.

![Screenshot 2024-04-08 at 8 08 01 PM](https://github.com/jinoh0731/Financial-news-information-retrieval/assets/111295393/1e44a974-0faf-4ce0-8f9c-b9f79635e447)

- **Performance**:Comparing `text-embedding-ada-002` to `text-embedding-3-small`, the average score on a commonly used benchmark for multi-language retrieval (MIRACL) has increased from 31.4% to 44.0%, while the average score on a commonly used benchmark for English tasks (MTEB) has increased from 61.0% to 62.3%.

- **Compact Design**: As indicated by the name, `text-embedding-3-small` is a smaller version of larger text embedding models. It is specifically optimized to maintain a balance between performance and resource efficiency.

- **Efficient Text Embeddings**: This model generates embeddings that are smaller in size but still capture the essential semantic features needed for tasks such as text similarity, classification, and clustering.

- **Speed and Efficiency**: The model's reduced size allows for faster processing times and lower memory usage, which is beneficial for applications running on limited resources or requiring real-time response.

- **Scalability**: Due to its efficiency and compact nature, `text-embedding-3-small` can easily scale to handle large volumes of text without significant resource expenditure.




# Model Performance
|                | Mini-LM                                        | Text-embedding-3                                    | BERT                                                                                                   | roBERTa                                         |
|----------------|------------------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Context Length | 512                                            | 8191                                                | 512                                                                                                    | 512                                             |
| Can Fine-Tune? | Yes                                            | No                                                  | Yes                                                                                                    | Yes                                             |
| Cost           | Free                                           | $0.02 / 1M tokens                                   | Free                                                                                                   | Free                                            |
| Source         | Microsoft                                      | OpenAI                                              | Google                                                                                                 | Facebook                                        |
| Permissions    | Open-Source on Hugging Face                    | OpenAI API Key                                      | Open-Source on Kaggle/Tensorflow Hub                                                                  | Open-Source on Hugging Face                     |
| Model Card Link| [Mini-LM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | [Text-embedding-3](https://platform.openai.com/docs/guides/embeddings/use-cases) | [BERT](https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/en-uncased-l-12-h-768-a-12/3) | [roBERTa](https://huggingface.co/FacebookAI/roberta-base) |




We evalulated model performance on the following metrics
* Success at K - A metric to establish whether we get a hit/relevant ESG article within K. Measures whether the relevant document (or item) appears in the top K positions of the model's ranking.
* Mean Reciprocal Rank (MRR) - MRR provides insight into the model's ability to return relevant items at higher ranks. It measures when does the first relevant ESG article appears. The closer this final number is to 1, the better the system is at giving you the right answers upfront.  
* Precision at K - Measures the proportion of retrieved documents that are relevant among the top K documents retrieved. It's calculated by dividing the number of relevant documents in the top K by K.
* Recall at K - Measures the proportion of relevant documents retrieved in the top K positions out of all relevant documents available. 
* F1 Score at K - Combines precision and recall into a single metric, offering a more comprehensive evaluation of the model's performance. It helps balance the trade-off between precision and recall, ensuring that neither is disproportionately favored.

Success at K:
| K | Mini-LM | Text-embedding-3 | Two Towers-BERT | Two Towers-roBERTa |
|---|---------|----------|-----------------|--------------------|
| 1 | 86.96%  | 88.52%   | 68.85%          | 70.49%             |
| 2 | 94.57%  | 100.00%  | 83.61%          | 90.16%             |
| 3 | 95.65%  | 100.00%  | 95.08%          | 95.08%             |
| 4 | 96.74%  | 100.00%  | 96.72%          | 98.36%             |
| 5 | 97.83%  | 100.00%  | 100.00%         | 100.00%            |

We want to select the K value where all models have achieved over 85% to conduct our model comparision. In our table above, we see that we should select K = 3.

| Model              | Precision at K=3 | Recall at K=3 | F1 at K=3 | MRR    |
|--------------------|------------------|---------------|-----------|--------|
| Mini-LM            | 79.71%           | 39.73%        | 45.56%    | 91.70% |
| Text-embedding-3   | 86.89%           | 27.37%        | 38.08%    | 94.26% |
| Two Towers-BERT    | 71.58%           | 21.54%        | 30.11%    | 81.12% |
| Two Towers-roBERTa | 79.23%           | 22.52%        | 33.21%    | 83.11% |


having more context length is impacting the model

# Code Demo

# Gradio Demo

# Limitation & Future Work for Improvement
## Limitation

#### 1. **Context Length Issue**
- **Problem**: Some encoder models used in the project, especially all BERT-related models, have limitations on the maximum length of text they can process (context length). This restricts the amount of information these models can consider, potentially omitting crucial details from longer documents.
- **Impact**: Important information may be truncated, leading to incomplete analysis of news articles.

#### 2. **Cost and Accessibility of Encoder Models**
- **Problem**: High-performing encoder model, in our case OpenAI Text Embedding, is not only costly to run but also lack the capability for fine-tuning in a cost-effective manner.
- **Impact**: This limits the project’s scalability and increases operational costs, making it less accessible for smaller asset management firms.

#### 4. **Two Tower Queries Similarity**
- **Problem**: Queries across different industries are too similar, reducing the effectiveness of industry-specific information retrieval.
- **Impact**: Reduces the model’s utility in providing industry-specific insights, which are critical for targeted investment decisions.

#### 5. **Subjectivity in Model Interpretations**
- **Problem**: The determination of what is considered ESG-relevant by GPT models is based on the prompts used, which may not align with investors' actual criteria or thoughts on what constitutes ESG.
- **Impact**: Potential misalignment with market perceptions could lead to irrelevant or overlooked insights.

#### 6. **Outdated Information**
- **Problem**: The dataset currently spans news articles from specific past dates (9/13/2022 to 9/8/2023). This approach does not capture recent events or developments, which are crucial for making informed investment decisions in real-time. Moreover, the dataset is skewed to have ESG related articles that reflect real news. For instance, significant events such as the bankruptcy of Silicon Valley Bank and the Ohio railroad derailment within the dataset's timeframe, which led to having significant amount of article related to those issues.
- **Impact**: Utilizing a dataset that doesn't reflect the most current events limits the tool's ability to provide actionable insights that reflect the latest market conditions. This reliance on historical data may lead to strategic missteps, particularly in rapid-response scenarios where current information could dramatically alter investment decisions. Additionally, the focus on certain major events may bias the model's understanding and responsiveness to other equally significant ESG-related developments.

#### 7. **Static Dataset**
- **Problem**: ESG investing and financial markets are highly sensitive to new information. This project only aims for building pipeline so that it could be utilized in many asset management or financial companies. Therefore, in order to utilize it in real-life, we would need another pipeline or method to web-scrape news articles in real-time and update database constantly to give the most current insights or reflect sudden market changes.
- **Impact**: The inability to process and analyze real-time data severely limits the system's effectiveness in a field where timeliness can significantly influence the success of investment strategies. Without the capability to update content dynamically, the tool may provide outdated or irrelevant recommendations, thereby diminishing its utility in fast-paced financial environments where current information is crucial for maximizing ESG investment returns.

## Recommendations for Improvement

#### 1. **Dynamic Data Acquisition**
- **Solution**: Implement a system that continuously scrapes and integrates live financial news articles. This would ensure that the dataset remains up-to-date and relevant, allowing asset managers to access the latest information.
- **Benefit**: Keeps the dataset current and maximizes the relevance of the information provided to asset managers, enhancing the accuracy and timeliness of investment decisions.

#### 2. **Automated ESG Criteria Updates**
- **Solution**: Regular updates to the SASB standards within our system to reflect the latest ESG criteria. This can be achieved by automating the scraping of SASB updates or integrating APIs that provide these updates.
- **Benefit**: Ensures continuous alignment with the latest ESG standards and investor expectations, thereby improving the reliability and relevance of the information retrieval system.

#### 3. **Enhanced Data Processing Capabilities**
- **Solution**: To handle the increased data volume from live scraping, enhancing our data processing and storage capabilities will be necessary. This may include more robust servers, faster processing algorithms, and more efficient data storage methods.
- **Benefit**: Supports scalable and efficient processing, enabling faster response times and handling larger data volumes without performance degradation. This improvement is crucial for maintaining high system performance even under increased load.

#### 4. **Exploration of Alternative Models**
- **Solution**: Test additional models that may offer better cost-efficiency, particularly those capable of handling longer context lengths or those offering more nuanced understanding of ESG topics.
- **Benefit**: Could discover more effective or efficient ways to process and analyze data, potentially reducing operational costs and improving insight accuracy. This exploration may also uncover innovative approaches that better match the specific needs of asset managers in ESG investing.

#### 5. **Model Fine-Tuning**
- **Solution**: Fine-tune existing models on newly acquired, industry-specific datasets to improve their predictive accuracy and relevance to current ESG criteria.
- **Benefit**: Fine-tuning models on specific datasets allows for more precise and tailored insights that can adapt to changes in market conditions or regulatory standards. This tailored approach enhances the model's utility and effectiveness in real-world applications, providing asset managers with more reliable and actionable data.

# Resource
Two Tower Resources:
- https://www.tensorflow.org/recommenders/examples/basic_retrieval

BERT: 
- https://www.tensorflow.org/text/tutorials/classify_text_with_bert
- https://www.kaggle.com/models/google/bert

RoBERTa
- https://huggingface.co/docs/transformers/en/model_doc/roberta

MiniLM Resources:
- https://arxiv.org/abs/2002.10957
- https://github.com/microsoft/unilm/tree/master/minilm
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

OpenAI Embedding Model:
- https://platform.openai.com/docs/guides/embeddings/embedding-models
- https://openai.com/blog/new-embedding-models-and-api-updates

GPT4 API:
- https://platform.openai.com/docs/models
- https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4


