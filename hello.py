from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define the context passage (text to be searched for answers)
context = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."

# Define the question to ask about the context
question = "Who was Albert Einstein?"

# Tokenize the question and context
inputs = tokenizer(question, context, return_tensors="pt")

# Perform inference (question answering)
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted answer span
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

# Print the question, context, and answer
print("Question:", question)
print("Context:", context)
print("Answer:", answer)
