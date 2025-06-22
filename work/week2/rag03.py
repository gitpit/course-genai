'''
rag03.py
# This code is a Python script that processes a text document, splits it into chunks, and allows for searching within those chunks using embeddings.
# It uses LangChain's RecursiveCharacterTextSplitter to split the text, HuggingFaceEmbeddings for generating embeddings, and Chroma as the vector store for similarity search.
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#         )
#         return text_splitter.split_documents(self.documents)
#         # The `split_documents` method uses the RecursiveCharacterTextSplitter to split the loaded documents into smaller chunks.    
'''

import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

class VisualizeChunking:
    def __init__(self, file_path, chunk_size=200, chunk_overlap=20):
        self.text = self.load_document(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self.split_document()
        self.embedding_model = self.get_embedding_model()
        self.vectorstore = self.create_vectorstore()
    
    def load_document(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
    
    def split_document(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
    
        return text_splitter.split_text(self.text)
    
    def get_embedding_model(self):
        return HuggingFaceEmbeddings()
    
    def create_vectorstore(self):
        return Chroma.from_texts(texts=self.docs, embedding=self.embedding_model)
    
    def view_splits(self):
        for i in range(len(self.docs) - 1):
            current_chunk = self.docs[i]
            next_chunk = self.docs[i + 1]
            
            print(f"\n\n{'='*80}")
            print(f"Chunk {i+1}/{len(self.docs)} - Length: {len(current_chunk)} chars")
            print(f"{'='*80}")
            
            overlap_text = ""
            for j in range(len(current_chunk)):
                if current_chunk[j:] == next_chunk[:len(current_chunk)-j]:
                    overlap_text = current_chunk[j:]
                    break
            
            if overlap_text:
                pre_overlap = current_chunk[:-len(overlap_text)]
                overlap_wrapped = textwrap.fill(" "+overlap_text, width=80)
                print_val = textwrap.fill(pre_overlap + overlap_wrapped, width=80)
                print(print_val, end="")
                print(textwrap.fill(f"\033[44m{overlap_wrapped}\033[0m", width=80), end='')
            else:
                print(textwrap.fill(current_chunk, width=80))
        
        print(f"\n{'='*80}")
        print(f"Chunk {len(self.docs)}/{len(self.docs)} - Length: {len(self.docs[-1])} chars")
        print(f"{'='*80}")
        print(textwrap.fill(self.docs[-1], width=80))
    
    def search(self, query, k=1):
        return self.vectorstore.similarity_search(query, k=k)
        

if __name__ == "__main__":
    workdir = os.path.dirname(os.path.abspath(__file__))
    docfile = workdir + r'\doc.txt'
    processor = VisualizeChunking(docfile)
    processor.view_splits()
    
    query1 = "How is coffee processed after harvesting?"
    result1 = processor.search(query1)
    print(result1)
    
    query2 = "Tell me about modern space exploration companies"
    result2 = processor.search(query2)
    print(result2)