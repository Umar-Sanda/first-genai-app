import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# Load model with caching
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    
    if torch.cuda.is_available():
        pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()  # Speed optimization
    else:
        pipe.to("cpu")
    
    return pipe

pipe = load_model()

# Streamlit UI
st.title("NIMBOU CLOUD Image Generator")
st.write("Enter a prompt below to generate an image")

# User input
prompt = st.text_input("Enter your prompt:", " ")
num_steps = st.slider("Number of Inference Steps", 10, 50, 30)  # Adjustable steps
generate_btn = st.button("Generate Image")

# Generate image
if generate_btn and prompt:
    with st.spinner("Generating... Please wait!"):
        image = pipe(prompt, num_inference_steps=num_steps).images[0]
        st.image(image, caption="Generated Image", use_container_width=True)
