'''
rag02_streamlit.py This code implements a book recommendation system using Streamlit and Sentence Transformers.
 It allows users to select a book from a predefined list and provides recommendations based on cosine similarity of text embeddings.

Streamlit and Gradio are similar—they both let you build interactive web apps in Python with minimal code.
  **Streamlit: Great for dashboards, data exploration, and custom layouts. You use widgets like sliders, buttons, and text boxes.
  **Gradio: Focused on quick demos for machine learning models, with built-in interfaces for text, images, audio, etc.

**Important Notes:
 - It works with v.env3.11
 - run it as streamlit run .\work\week2\rag02_streamlit.py
'''

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Book Recommendation System")

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

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2") # this model is trained with a dataset from various sources including Wikipedia, web texts, 
                            #and books and is optimized for generating sentence embeddings that capture semantic meaning. I verify that this statemment is correct.


@st.cache_data
def get_embeddings(texts):
    model = load_model()
    return model.encode(texts)

texts = [f'{book["title"]}. {book["content"]}' for book in books]
embeddings = get_embeddings(texts)

st.subheader("Select a book to get recommendations")

book_titles = [book["title"] for book in books]
selected_title = st.selectbox("Choose a book", book_titles)
button_clicked = book_titles.index(selected_title) if st.button("Get Recommendations") else None

if button_clicked is not None:
    selected_book = books[button_clicked]
    st.write(f"### You selected: {selected_book['title']}")
    st.write(selected_book["content"])
    
    similarities = cosine_similarity([embeddings[button_clicked]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:3]
    recommendations = [books[idx] for idx in similar_indices]
    
    st.write("### Recommended books:")
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. **{rec['title']}**")
        st.write(f"   {rec['content']}")

print("!It works!!")