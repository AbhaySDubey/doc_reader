from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import gradio as gr

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
    
def post_process_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove repeated phrases
    phrases = text.split('. ')
    unique_phrases = list(dict.fromkeys(phrases))
    text = '. '.join(unique_phrases)
    return text

def ocr(image):
    queries = [
        "Extract and transcribe all the text visible in the image, including any small or partially visible text.",
        # "Look closely at the image and list any text you see, no matter how small or unclear.",
        # "What text can you identify in this image? Include everything, even if it's partially obscured or in the background."
    ]

    all_extracted_text = []
    for query in queries:
        extracted_text = extract_text(image, query)
        all_extracted_text.append(extracted_text)

    # Combine and deduplicate the results
    final_text = "\n".join(set(all_extracted_text))

    final_text = post_process_text(final_text)
    return final_text
    
    
def main_fun(image, keyword):
    ext_text = ocr(image)
    
    if keyword:
        highlight_text = re.sub(f'({re.escape(keyword)})', r'<span style="background-color: yellow;">\1</span>', ext_text, flags=re.IGNORECASE)
    else:
        highlight_text = ext_text
    
    return ext_text, highlight_text

iface = gr.Interface(
    fn=main_fun,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Textbox(label="Enter search term", placeholder="Search")
    ],
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.HTML(label="Search Results")
    ],
    title="Document Search using OCR (English/Hindi)"
)

iface.launch()