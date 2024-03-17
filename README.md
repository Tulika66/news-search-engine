# Overview

This project implements a basic news search engine using the Vector Space Model (VSM) approach. 

Here's an explanation of its purpose and functionality:

* **Data Preprocessing**: The code first preprocesses a collection of news articles obtained from the BBC news dataset. It tokenizes the text, applies stemming using the Porter stemmer, and removes stopwords.

* **Term Frequency (TF)**: It calculates the term frequency for each word in the corpus and stores it in a DataFrame. Term frequency is the number of times a term appears in a document.

* **Inverse Document Frequency (IDF)**: It calculates the inverse document frequency for each term, which measures the informativeness of the term across the entire corpus.

* **TF-IDF Calculation**: The code computes the TF-IDF (Term Frequency-Inverse Document Frequency) scores for each term in each document. TF-IDF is a numerical statistic that reflects the importance of a term in a document relative to a collection of documents.

* **Document Similarity Calculation**: It calculates the cosine similarity between the query and each document in the corpus using the TF-IDF scores. Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity.

* **Ranking and Displaying Results**: Finally, the code ranks the documents based on their similarity to the query and displays the top-ranked documents along with their content.

The purpose of this news search engine is to allow users to input a query, such as a search term or phrase, and retrieve relevant news articles from the dataset based on the similarity between the query and the content of the articles.
