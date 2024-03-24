Goal is to create RAG system with PostgreSQL and PGVector.

As of now, we have done following tasks.
1. Load document and split it into chunks
2. create embeddings of 384 dimensions using open source SentenceTransformer 'multi-qa-MiniLM-L6-cos-v1' model.
3. store these embeddings into the PostgreSQL database using pg_vector

TODO
4. Run similarity search - PARTIALY COMPLETE
5. Run the complete RAG system
6. Explore indexing in pg_vector for performance


