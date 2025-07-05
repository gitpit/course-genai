'''
#rag01.py This code demonstrates how to use the GloVe model from Gensim to perform word vector operations
# with a dataset of input-output pairs. It includes a custom dataset class, a simple neural
# network model, and a trainer class for managing the training process. The code also includes examples of how to 
# perform word vector operations such as cosine similarity and analogy tasks using the GloVe model.

'''

import numpy as np
import gensim.downloader as api
#gensim is a library for topic modeling and document similarity; 
# gensim.downloader is used to download pre-trained models

model = api.load("glove-wiki-gigaword-50") #what is this model?
# This line loads the GloVe model with 50-dimensional word vectors trained on Wikipedia and Gigaword corpus.
model["banana"] #what is this?
# This retrieves the word vector for "banana" from the GloVe model.
v_banana = model["banana"]
v_grape = model["grape"]
v_aeroplane = model["aeroplane"]

sim_banana_all = model.cosine_similarities(v_banana, [v_banana, v_grape, v_aeroplane]) #what is this?
# This calculates the cosine similarity between the vector for "banana" and the vectors for "banana", "grape", and "aeroplane".
# Cosine similarity measures how similar two vectors are, with values closer to 1 indicating more similarity.
print(sim_banana_all)
# The output will show the cosine similarity scores for "banana" with respect to itself, "grape", and "aeroplane".
# The expected output will be an array of similarity scores, e.g., [1.0, 0.5, 0.3], where 1.0 is the similarity with itself.

v_king = model["king"]
v_queen = model["queen"]
v_man = model["man"]
new_word = model.most_similar(v_man+v_queen-v_king, topn=1)
# This line finds the word that is most similar to the vector obtained by adding "
#why v_queen - v_king
#man" and "queen" and subtracting "king".
# This operation is often used to find analogies, such as "
#man is to queen as ? is to king".
# The expected output will be a list containing the most similar word and its similarity score, e
v_paris = model["paris"]
v_france = model["france"]
v_india = model["india"]

new_word2 = model.most_similar(v_paris - v_france +v_india, topn=1)


# what is chroma db which is like gensim?
# ChromaDB is a vector database designed for efficient storage and retrieval of high-dimensional vectors, often used in machine learning applications.

