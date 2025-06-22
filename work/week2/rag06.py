'''
rag06.py
# This code demonstrates how to use a CrossEncoder model from the Sentence Transformers library to rank documents based on their relevance to a given query.
#             print(f"Overlap with next chunk: {overlap_text}")
# It defines a query and a list of documents, then uses the CrossEncoder to score each document against the query, ranking them accordingly.
#             print(f"Current Chunk:\n{textwrap.fill(current_chunk, width=80)}")
            from sentence_transformers import SentenceTransformer, util
'''
from sentence_transformers import CrossEncoder #what is this?
# CrossEncoder is a model from the Sentence Transformers library that is used to score pairs of sentences (or a query and a document) based on their relevance.
# It takes a query and a document as input and outputs a score indicating how relevant the document is to the query.

query = "What is the capital of France?"
documents = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital and largest city of Germany.",
    "France is a country in Western Europe with several overseas territories.",
    "The Eiffel Tower is located in Paris, France.",
    "Lyon is the third-largest city in France after Paris and Marseille.",
    "France has a population of approximately 67 million people.",
    "The capital of Italy is Rome, which is known as the Eternal City.",
    "Paris hosts many famous landmarks including the Louvre Museum."
]

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
query_doc_pairs = [[query, doc] for doc in documents]
cross_encoder_scores = cross_encoder_model.predict(query_doc_pairs)
ranked_documents = [doc for _, doc in sorted(zip(cross_encoder_scores, documents), reverse=True)]
