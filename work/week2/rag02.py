import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

books = [
  {
    "title": "Introduction to Probability",
    "content": ("Joseph K. Blitzstein and Jessica Hwang's textbook (also known as 'the Harvard stats 110 book') "
                "provides an accessible introduction to probability theory with engaging examples and exercises.")
  },
  {
    "title": "A First Course in Probability",
    "content": ("Sheldon Ross's widely-used textbook covers fundamental probability concepts, random variables, "
                "expectation, and stochastic processes with mathematical rigor and practical examples.")
  },
  {
    "title": "Linear Algebra and Its Applications",
    "content": ("Gilbert Strang's classic textbook combines theoretical foundations with practical applications, "
                "making it one of the most popular introductions to linear algebra for engineers and scientists.")
  },
  {
    "title": "Linear Algebra Done Right",
    "content": ("Sheldon Axler's approach to linear algebra emphasizes conceptual understanding by avoiding "
                "determinants and focusing on vector spaces and linear transformations.")
  },
  {
    "title": "Pattern Recognition and Machine Learning",
    "content": ("Christopher Bishop's landmark textbook presents a comprehensive introduction to machine "
                "learning, combining mathematical depth with practical applications and algorithms.")
  },
  {
    "title": "Designing Machine Learning Systems",
    "content": ("Chip Huyen's practical guide explores the challenges of building machine learning systems "
                "that can be deployed reliably in production environments.")
  },
  {
    "title": "The LLM Engineer's Handbook",
    "content": ("This comprehensive guide covers the practical aspects of working with and deploying "
                "large language models, including prompt engineering, fine-tuning, and evaluation methods.")
  },
  {
    "title": "Illustrating Large Language Models",
    "content": ("Jay Alammar's visual guide breaks down complex concepts in LLMs through intuitive "
                "illustrations, making transformer architectures and attention mechanisms accessible.")
  },
  {
    "title": "System Design Interview - An Insider's Guide",
    "content": ("Alex Xu's guide helps software engineers prepare for system design interviews by walking "
                "through common design scenarios and providing detailed approaches to solving them.")
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2") #what is this model?
# This line loads a pre-trained SentenceTransformer model that is optimized for generating sentence embeddings. 
# It is a smaller version of the MiniLM model, designed for efficient computation while maintaining good performance on various NLP tasks.
# The model is used to convert sentences or texts into fixed-size vector representations (embeddings) that capture their semantic meaning.
# The embeddings can then be used for tasks like similarity comparison, clustering, or classification.
# The model "all-MiniLM-L6-v2" is a lightweight transformer model that provides efficient and effective sentence embeddings.


texts = [f'{book["title"]}. {book["content"]}' for book in books]

embeddings = model.encode(texts) #what is this?
# This line generates embeddings for the provided texts using the pre-trained SentenceTransformer model.    
# The `encode` method processes each text and converts it into a fixed-size vector representation (embedding).
# These embeddings capture the semantic meaning of the texts, allowing for comparison and analysis of their content.    
# The resulting `embeddings` variable is a NumPy array where each row corresponds to the embedding of a text in the `texts` list.
# The embeddings are used to measure similarity between texts, enabling tasks like recommendation or clustering based on content similarity.    


book_id = 0
similarities = cosine_similarity([embeddings[book_id]], embeddings)[0] #what is this line?
# This line calculates the cosine similarity between the embedding of the book at index `book_id` and all other embeddings.
# The `cosine_similarity` function computes the cosine of the angle between two vectors, which  
# is a measure of similarity. A cosine similarity of 1 indicates that the vectors are identical, while a value closer to 0 indicates less similarity.
# The result is a NumPy array where each element represents the similarity score between the selected book and all other books.
# The `[0]` at the end extracts the first row of the resulting similarity matrix,
# which corresponds to the similarities of the selected book with all other books in the dataset.
# The `similarities` array now contains the similarity scores for the selected book against all other books.
similar_indices = np.argsort(similarities)[::-1][1:3] #what is this line?
# This line sorts the indices of the `similarities` array in descending order and selects the top 2 indices (excluding the first one, which is the book itself).
# The `np.argsort(similarities)` function returns the indices that would sort the `similarities` array in ascending order.
# The `[::-1]` reverses the order to get descending indices, and `[1:3]` slices the array to get the second and third most similar books (excluding the first index, which corresponds to the book itself).
# The resulting `similar_indices` array contains the indices of the two books that are most similar to the book at `book_id`.
# These indices can be used to retrieve the titles or contents of the most similar books for further analysis or recommendations.
recommendations = [books[idx] for idx in similar_indices]