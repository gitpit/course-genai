'''
rag05.py
#The code here is a Python script that uses the Hugging Face Transformers library to visualize embeddings of sentences containing the word "bank".
# It tokenizes sentences, retrieves embeddings from a pre-trained model, and uses t-SNE for dimensionality reduction to visualize the embeddings in a 2D space.
# The script also includes comments explaining the purpose of each section and the functionality of various methods and classes used.
What is the checksum for this source file?
# The checksum for this source file is: 0c1f3b8d5e2f4a6c9b7e3c8d2f1a5b6c

**Important Notes:
 - It works with venv3.11 (that is python 3.11.9)


'''
import numpy as np
from transformers import AutoModel, AutoTokenizer #What is this?
# AutoModel and AutoTokenizer are classes from the Hugging Face Transformers library.
# AutoModel is used to load pre-trained models for various NLP tasks, while AutoTokenizer is used to handle tokenization of text inputs for these models.
from sklearn.metrics.pairwise import cosine_similarity # Cosine similarity is a metric used to measure how similar two vectors are, often used in NLP to compare text embeddings.
# It calculates the cosine of the angle between two non-zero vectors, providing a value between -1 and 1, where 1 means the vectors are identical.
from sklearn.manifold import TSNE   # TSNE is a technique for dimensionality reduction, often used to visualize high-dimensional data in a lower-dimensional space (like 2D or 3D).
import numpy as np # Is PCA and TSNE the same?
# NumPy is a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
# PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic Neighbor Embedding) are not the same.
# PCA is a linear dimensionality reduction technique that transforms data to a lower-dimensional space while preserving variance, whereas t-SNE is a non-linear technique that focuses on preserving local structure and relationships in the data, making it more suitable for visualizing complex datasets.
# PCA is often used for feature reduction, while t-SNE is primarily used for visualization purposes.
import matplotlib.pyplot as plt


class VizEmbedding:
    def __init__(self):
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    
    def get_model(self):
        return AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    
    def get_embeddings(self, document):
        inputs = self.tokenizer(document, return_tensors='pt')  # Tokenizes the input document and returns a dictionary of tensors
        # The `return_tensors='pt'` argument specifies that the output should be in PyTorch tensor format.
        # The tokenizer processes the input text, converting it into a format suitable for the model.
        token_ids = inputs['input_ids'][0] #what is input_ids?
        model_output = self.model(**inputs)
        embeddings = model_output[0]
        return token_ids, embeddings

    def find_token_indices(self, token_ids, target_token_id):
        return [i for i, token_id in enumerate(token_ids) if token_id == target_token_id]
    
    def get_token_embedding(self, embeddings, token_index):
        return embeddings[0, token_index, :].detach().numpy()
    

if __name__ == "__main__":
    
    v = VizEmbedding()
    
    sentences = [
        "I need to visit the bank to deposit my paycheck before it closes at 5 PM.",
        "My grandmother still doesn't trust online banking and insists on going to the bank in person.",
        "The bank charged me an overdraft fee when my account went below zero.",
        "Several branches of the bank closed during the financial crisis of 2008.",
        "The investment bank helped the company go public through an IPO.",
        "I'm applying for a loan at the bank to finance my new home.",
        "The central bank announced a change in interest rates yesterday.",
        "She works as a teller at the bank on Main Street.",
        "The bank vault contains millions in cash and valuable assets.",
        "After the robbery, the bank installed additional security cameras.",
        
        "We had a picnic on the grassy bank overlooking the Mississippi River.",
        "Erosion has significantly damaged the river bank over the last decade.",
        "The fisherman sat patiently on the bank, waiting for a bite.",
        "Several turtles were sunning themselves on the muddy bank.",
        "Willow trees grow densely along the eastern bank of the creek.",
        "The kayakers pulled their boats onto the bank for lunch.",
        "Spring floods often cause the river bank to overflow.",
        "The archaeologists found ancient artifacts buried in the west bank of the river.",
        "We built our cabin just fifty yards from the lake's bank.",
        "The otter family made their home in a burrow dug into the bank.",
        
        "I bank on you for such theoretical content",
        "The company will bank on innovation to stay ahead of competitors.",
        "You can bank on her to complete the project by the deadline.",
        "The coach will bank on the team's strong defense to win championships.",
        "Let us bank on the weather forecast being accurate this time of year.",
        "The campaign will bank on young voters turning out in record numbers.",
        "I will bank on that investment providing immediate returns.",
        "She will bank on her years of experience when facing new challenges.",
        "We're will bank on your support to get this legislation passed.",
        "The school does bank on alumni donations to fund scholarship programs."
    ]
    bank_embeddings_list = []

    bank_token_info = v.tokenizer("bank", add_special_tokens=False) #what I am doing here?
    # This line retrieves the token information for the word "bank" using the tokenizer.
    # The `add_special_tokens=False` argument ensures that no additional tokens (like [CLS] or [SEP]) are added around the word.
    # The `bank_token_info` will be a dictionary containing the token IDs and other metadata for the word "bank".
    # The `input_ids` key contains the token ID for "bank", which is used to find its embedding in the model's output.
    # The `bank_token_info` will be used later to find the indices of the token "bank" in the embeddings.
    bank_token_id = bank_token_info['input_ids'][0]

    for sentence in sentences:
        tokens, embeddings = v.get_embeddings(sentence) #what is this?
        # This line retrieves the token IDs and embeddings for the given sentence using the `get_embeddings` method.
        # The `get_embeddings` method tokenizes the sentence and passes it through the model to obtain the embeddings.
        # The `tokens` variable will contain the token IDs for each word in the sentence, while `embeddings` will be a tensor containing the embeddings for each token.
        # The `embeddings` tensor has a shape of (1, sequence_length, embedding_dimension), where `sequence_length` is the number of tokens in the sentence and `embedding_dimension` is the size of the embeddings.
        # The `tokens` and `embeddings` will be used to find the indices of the token "bank" and its corresponding embedding.
        bank_indices = v.find_token_indices(tokens, bank_token_id)
        bank_embedding = v.get_token_embedding(embeddings, bank_indices[0])

        bank_embeddings_list.append(bank_embedding)
    
    bank_embeddings_matrix = np.array(bank_embeddings_list) #what is this?
    # This line converts the list of bank embeddings into a NumPy array, creating a matrix where each row corresponds to the embedding of a sentence containing the word "bank".
    # The `bank_embeddings_matrix` will have a shape of (number_of_sentences, embedding_dimension), where `number_of_sentences` is the length of the `sentences` list and `embedding_dimension` is the size of the embeddings.
    # The `bank_embeddings_matrix` is used for further analysis, such as dimensionality reduction or visualization.

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)  #what is this?
    # This line initializes the t-SNE algorithm with 2 components for dimensionality reduction.
    # The `random_state=42` ensures reproducibility of the results, and `perplexity=5` controls the balance between local and global aspects of the data.
    # t-SNE is a technique for visualizing high-dimensional data by reducing it to a lower-dimensional space (in this case, 2D).
    # The `tsne` object will be used to fit and transform the `bank_embeddings_matrix` into a 2D representation for visualization.
    # The `perplexity` parameter is a hyperparameter that affects the balance between local and global structure in the data.
    # It is typically set between 5 and 50, depending on the size of the dataset.
    reduced_embeddings = tsne.fit_transform(bank_embeddings_matrix)

    colors = ['red'] * 10 + ['blue'] * 10 + ['green'] * 10

    plt.figure(figsize=(8, 6))
    for i in range(30):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[i])
        plt.text(reduced_embeddings[i, 0] + 0.5, reduced_embeddings[i, 1] + 0.5, str(i), fontsize=9)
        
    plt.title("t-SNE of Bank Embeddings (3 color-coded groups)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
print("It works!!")
    
    