import gradio as gr
from huggingface_hub import InferenceClient
import os

model_id = "VoltIC/Automated-Text-Summarizer"
client = InferenceClient(model=model_id, token=os.getenv("HF_TOKEN"))

def summarize_text(text):
    input_len = len(text.split())
    try:
        summary = client.summarization(text)
        output_len = len(summary.split())
        
        # Calculate reduction %
        reduction = round((1 - output_len/input_len) * 100)
        
        return f"{summary}\n\n---\n📊 Compression: {reduction}% (Reduced from {input_len} to {output_len} words)"
    except Exception as e:
        return f"Error: {e}"

# 2. Simplified Interface to avoid the IndexError
with gr.Blocks() as app:
    gr.Markdown("# Aditya's Instant Summarizer")
    gr.Markdown("Uses the HF Inference API to avoid large downloads.")
    
    input_box = gr.Textbox(lines=8, label="Input Article")
    output_box = gr.Textbox(label="Summary")
    submit_btn = gr.Button("Summarize")
    
    submit_btn.click(fn=summarize_text, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    app.launch()