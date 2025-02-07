import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch

# Load BLIP model and processor
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Load MarianMT translation model for English to Malayalam
    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
    translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
    
    return processor, model, translator_model, translator_tokenizer

processor, model, translator_model, translator_tokenizer = load_models()

# Function to generate image explanation in English
def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to("cpu")  # Use CPU (change to "cuda" for GPU)
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Function to translate English caption to Malayalam
def translate_to_malayalam(english_text):
    inputs = translator_tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

# Streamlit UI
st.title("🖼️ Image Explanation App")
st.write("Upload an image, and the AI will describe it in English or Malayalam.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (PNG, JPEG, JPG)", type=["png", "jpg", "jpeg"])

# Language selection
language = st.radio("Select caption language:", ["English", "Malayalam"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating explanation..."):
        caption = generate_caption(image)
        if language == "Malayalam":
            caption = translate_to_malayalam(caption)

    st.subheader("AI Explanation:")
    st.write(caption)
