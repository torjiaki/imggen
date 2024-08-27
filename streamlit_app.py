import streamlit as st
from PIL import Image
import
from transformers import StableDiffusionPipeline

# Load the pre-trained model
@st.cache_resource
def load_model():
    # You can change this to any other model you'd like to use
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return model

model = load_model()

def generate_image(prompt, init_image):
    # Process the input image
    init_image = init_image.convert("RGB")
    init_image = init_image.resize((512, 512))  # Resize for the model's input size
    
    # Generate the new image
    with torch.no_grad():
        generated_images = model(prompt=prompt, init_image=init_image, strength=0.75).images
    return generated_images[0]

st.title("AI Image Generation")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    prompt = st.text_input("Enter a prompt for image generation:", "A beautiful landscape")

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            input_image = Image.open(uploaded_image)
            generated_image = generate_image(prompt, input_image)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
