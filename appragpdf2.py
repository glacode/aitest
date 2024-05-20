from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.spatial.distance import cdist

# Load document data (replace with your data)
documents = "The capital of France is Paris. Mount Everest is the tallest mountain in the world. Whales are mammals, not fish."

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # Assuming you have this function defined
# chunks = text_splitter.split_documents(documents)
chunks = text_splitter.split_text(documents)

# Load sentence transformer model (CPU-based)
# model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
model = SentenceTransformer('quora-distilbert-base', device='cpu')

# Embed document chunks and query
def embed(text):
  return model.encode(text)

# document_embeddings = [embed(chunk['text']) for chunk in chunks]
document_embeddings = [embed(chunk) for chunk in chunks]

# Load question-answering model (local)
# qa_model_name = "distilbert-qa-base-uncased"  # Replace with your model name
qa_model_name = "distilbert-base-uncased-distilled-squad"  # Replace with your model name
# qa_pipeline = pipeline("question-answering", model=qa_model_name, device=0)  # Assuming CPU on device 0
qa_pipeline = pipeline("question-answering", model=qa_model_name, device='cpu')  # Assuming CPU on device 0

def answer_question(query):
  # Embed query
  query_embedding = embed(query)

  # Calculate cosine distances
  distances = cdist(query_embedding.reshape(1, -1), document_embeddings, metric="cosine")

  # Retrieve top k closest documents (indices)
  top_k_indices = distances.argsort(axis=1)[0][:3]  # Get top 3 closest

  # Prepare context for Q/A model
  # context = " ".join([chunks[i]['text'] for i in top_k_indices])
  context = " ".join([chunks[i] for i in top_k_indices])

  # Generate answer with local Q/A model
  answer = qa_pipeline(question=query, context=context)["answer"]
  
  return answer

# Example usage
question = "What is the capital of France?"
answer = answer_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
