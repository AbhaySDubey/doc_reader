from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
import gradio as gr
import re

rag = RAGMultiModalModel.from_pretrained("vidore/colpali")
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

def extract_text(image, query):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cpu")
    with torch.no_grad():
        generated_ids = vlm.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def search_text(text, query):
    if query:
        searched_text = re.sub(f'({re.escape(query)})', r'<span style="background-color: yellow;">\1</span>', text, flags=re.IGNORECASE)
    else:
        searched_text = text
    return searched_text

def extraction(image, query):
    extracted_text = extract_text(image, query)
    return extracted_text, extracted_text   # return twice - one to display output and the other for state management


"""
    Main App
"""
with gr.Blocks() as main_app:
    gr.Markdown("# Document Reader using OCR(English/Hindi)")
    gr.Markdown("### Use Doc_Reader to extract text out of documents - images(OCR) or ask questions based on the input image")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload an Image")
            
            gr.Markdown("""
                        ### Please use this prompt for text extraction 
                        **What text can you identify in this image? Include everything, even if it's partially obscured or in the background.**
                        """)
            query_input = gr.Textbox(label="Enter query for retrieval", placeholder="Query/Prompt")
            extract_button = gr.Button("Read Doc!")
            
            search_input = gr.Textbox(label="Enter search term", placeholder="Search")
            search_button = gr.Button("Search!")
            
            
        with gr.Column():
            extracted_text_op = gr.Textbox(label="Output")
            search_text_op = gr.HTML(label="Search Results")
            
            download_button = gr.Button("Download Plain Text")
        
        # Retrieval
        extracted_text_state = gr.State()
        extract_button.click(
            extraction,
            inputs=[img_input, query_input],
            outputs=[extracted_text_op, extracted_text_state]
        )
        
        # Search
        search_button.click(
            search_text,
            inputs=[extracted_text_state, search_input],
            outputs=[search_text_op]
        )
        
        # Download
        download_button.click(
            lambda text: gr.File.save_text_to_file(text, "extracted_text.txt"),
            inputs=[extracted_text_state],
            outputs=[gr.File(label="Download Extracted Text")]
        )
        
main_app.launch()