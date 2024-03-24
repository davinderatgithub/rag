from pypdf import PdfReader

def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""

    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Check if text is extracted successfully
            text += extracted_text + "\n"  # Append text of each page

    return text

# Example usage
pdf_text = read_pdf("/Users/davindersingh/mywork/rag/new_chapter_Beautiful_Lesson.pdf")
#print(pdf_text)


# def split_text(text, chunk_size=1000, overlap=50):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunks.append(text[start:end])
#         start = end - overlap

#     return chunks

import concurrent.futures

def split_text(text, chunk_size=500, overlap=50):
    chunks = []

    def process_chunk(start):
        end = min(start + chunk_size, len(text))
        return text[start:end]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        starts = range(0, len(text), chunk_size - overlap)
        chunks = list(executor.map(process_chunk, starts))

    return chunks

# Example usage
#print("\n Started split.")
#text_chunks = split_text(pdf_text)
# print("\n=======================================\n")
# print(text_chunks[0])


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
#dimentions = 384 

# query_embedding = model.encode('How big is London')
# passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
#                                   'London is known for its finacial district'])

# query_embedding = model.encode('breakup')


#embeddings = model.encode(text_chunks)

#print("Similarity:", util.dot_score(query_embedding, passage_embedding))

# Storing Embeddings in PostgreSQL
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import inspect
from sqlalchemy import func
import numpy as np

Base = declarative_base()
N_DIM = 384

class TextEmbedding(Base):
    __tablename__ = 'text_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(String)
    embedding = Column(Vector(N_DIM))

# Connect to PostgreSQL
# for postgreSQL database credentials can be written as 
user = 'davindersingh'
password = ''
host = 'localhost'
port = '5432'
database = 'postgres'
# for creating connection string
connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
# SQLAlchemy engine
engine = create_engine(connection_str)
# you can test if the connection is made or not
try:
    with engine.connect() as connection_str:
        print('Successfully connected to the PostgreSQL database')
except Exception as ex:
    print(f'Sorry failed to connect: {ex}')

# The recommended way to check for existence
if not inspect(engine).has_table("text_embeddings"):
    Base.metadata.create_all(engine)
else:
    print('text_embeddings table already exists.')

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# insert embeddings
def insert_embeddings(embeddings):
    for i, embedding in enumerate(embeddings):
        new_embedding = TextEmbedding(content=text_chunks[i], embedding=embedding)
        session.add(new_embedding)
    session.commit()

#insert_embeddings(embeddings)

# Query all rows from the TextEmbedding table
#rows = session.query(TextEmbedding).all()

def find_similar_embeddings(query_embedding, limit=5):
    k = 10
    similarity_threshold = 0.01
    query = session.query(TextEmbedding, TextEmbedding.embedding.cosine_distance(query_embedding).label("distance")).filter(TextEmbedding.embedding.cosine_distance(query_embedding) < similarity_threshold).order_by("distance").limit(k).all()
    return query

query = "Last night I was with my college"
query_embedding = model.encode(query)
# Call the function and print the output
print("Started similarity search")
#results = find_similar_embeddings(query_embedding, limit=5)
#for result in results:
#    print(f"Content: {result.TextEmbedding.content}, Distance: {result.distance}")

def find_similar_embeddings_exact(query_embedding, limit=5):
    results = session.query(TextEmbedding).filter(TextEmbedding.content == query_embedding).limit(limit).all()
    return results

results = find_similar_embeddings_exact(query, limit=5)
for result in results:
    print(f"Content: {result.TextEmbedding.content}")
