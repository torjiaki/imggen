import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
from io import BytesIO

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    model.to("cuda")
    return model

model = load_model()

def generate_image(prompt, init_images):
    # Process input images
    processed_images = [image.convert("RGB").resize((512, 512)) for image in init_images]
    
    # Generate the new images
    generated_images = []
    for img in processed_images:
        with torch.no_grad():
            generated_image = model(prompt=prompt, init_image=img, strength=0.75).images[0]
            generated_images.append(generated_image)
    return generated_images

st.title("AI Image Generation from Up to 5 Input Images")

uploaded_files = st.file_uploader("Choose up to 5 images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload up to 5 images only.")
    else:
        # Display uploaded images
        st.write("Uploaded Images:")
        input_images = [Image.open(file) for file in uploaded_files]
        for img in input_images:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        prompt = st.text_input("Enter a prompt for image generation:", "A beautiful landscape")

        if st.button("Generate Images"):
            with st.spinner("Generating images..."):
                generated_images = generate_image(prompt, input_images)
                for idx, img in enumerate(generated_images):
                    st.image(img, caption=f"Generated Image {idx + 1}", use_column_width=True)
