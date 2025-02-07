import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

def get_image_explanation(image):
    inputs = processor(image, return_tensors="pt").to("cpu") 
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

st.title("Image Caption App")
st.write("Upload an image, and the AI will describe it in English.")

uploaded_file = st.file_uploader("Upload an image (PNG, JPEG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating explanation..."):
        explanation = get_image_explanation(image)

    st.subheader("AI Explanation:")
    st.write(explanation)
