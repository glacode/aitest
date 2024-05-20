from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.spatial.distance import cdist

import PyPDF2

def read_pdf(filepath: str):
  try:
    # Open PDF file
    with open(filepath, 'rb') as pdf_file:
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      num_pages = len(pdf_reader.pages)

      # Extract text from all pages
      text = ""
      for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
      return text

  except FileNotFoundError:
    print("Error: PDF file not found.")
    return ""

# Load document data (replace with your data)
# documents = "The capital of France is Paris. Mount Everest is the tallest mountain in the world. Whales are mammals, not fish."
documents = read_pdf("/home/mionome/Desktop/fremlin-mt1.pdf")

# Split documents into chunks
text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
  )
    
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
# question = "What is the capital of France?"
question = "What is a measure space?"
answer = answer_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
