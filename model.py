from transformers import pipeline

model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)

# Save the model and tokenizer to a local directory
# This is the Hugging Face standard for 'pickling' a model
summarizer.save_pretrained("./summarizer_model")
print("Model saved successfully to ./summarizer_model")