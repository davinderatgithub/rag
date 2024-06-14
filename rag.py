from itertools import chain
import torch
from pgvector.psycopg2 import register_vector

from db import get_connection
from embedding import generate_embeddings

from pgvector.psycopg2 import register_vector

template = """<s>[INST]
You are a friendly documentation search bot.
Use following piece of context to answer the question.
If the context is empty, try your best to answer without it.
Never mention the context.
Try to keep your answers concise unless asked to provide details.

Context: {context}
Question: {question}
[/INST]</s>
Answer:
"""

def get_retrieval_condition(query_embedding, threshold=0.7):
    # Convert the query embedding to a comma-separated string
    query_embedding_str = ",".join(map(str, query_embedding))

    # Create the SQL condition
    condition = f"(embeddings <=> '{query_embedding_str}') < {threshold}"
    condition += " ORDER BY embeddings <=> '{query_embedding_str}'"

    return condition


def rag_query(tokenizer, model, device, query):
    # Generate query embedding
    query_embedding = generate_embeddings(tokenizer, model, device, query)[1]

    # Retrieve relevant embeddings from the database
    retrieval_condition = get_retrieval_condition(query_embedding)
    retrieved = execute_sql_query(retrieval_condition)

    # Concatenate the retrieved fragments
    rag_query = ' '.join([row[0] for row in retrieved])

    # Format the query template
    query_template = template.format(context=rag_query, question=query)

    # Encode the query template
    input_ids = tokenizer.encode(query_template, return_tensors="pt")

    # Generate the response
    generated_response = model.generate(input_ids.to(device), max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated_response[0][input_ids.shape[-1]:], skip_special_tokens=True)

def execute_sql_query(retrieval_condition):
    conn = get_connection()
    register_vector(conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT doc_fragment FROM embeddings WHERE {retrieval_condition} LIMIT 5")
    return cursor.fetchall()
