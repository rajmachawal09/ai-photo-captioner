import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForImageClassification, VisionEncoderDecoderModel

@st.cache_resource
def load_models():
    try:
        st.info("Loading models... This may take a few minutes.")
        image_classifier = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
        caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        caption_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        st.success("Models loaded successfully!")
        return image_classifier, image_processor, caption_model, caption_processor, caption_tokenizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        st.error(f"Error fetching image: {str(e)}")
        return None

def analyze_image(image, image_classifier, image_processor):
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = image_classifier(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return image_classifier.config.id2label[predicted_class_idx]

def generate_caption(image, caption_model, caption_processor, caption_tokenizer):
    inputs = caption_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    
    max_length = 50
    
    with torch.no_grad():
        output_ids = caption_model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def generate_hashtags(image_content, caption):
    words = set(image_content.lower().split() + caption.lower().split())
    return [f"#{word}" for word in words if len(word) > 3 and not word.isdigit()]

def process_image(image_url, models):
    image_classifier, image_processor, caption_model, caption_processor, caption_tokenizer = models
    
    try:
        image = get_image_from_url(image_url)
        if image is None:
            return {"error": "Failed to fetch image"}
        
        image_content = analyze_image(image, image_classifier, image_processor)
        caption = generate_caption(image, caption_model, caption_processor, caption_tokenizer)
        hashtags = generate_hashtags(image_content, caption)
        
        return {
            "content": image_content,
            "caption": caption,
            "hashtags": hashtags,
            "image": image
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def main():
    st.title("Image Analysis and Caption Generator")

    models = load_models()
    if any(model is None for model in models):
        st.error("Failed to load models. Please try again later.")
        return

    image_url = st.text_input("Enter the URL of the image you want to analyze:")
    
    if st.button("Analyze Image"):
        if image_url:
            with st.spinner("Analyzing image..."):
                result = process_image(image_url, models)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.image(result["image"], caption="Uploaded Image", use_column_width=True)
                st.subheader("Image Content:")
                st.write(result["content"])
                st.subheader("Generated Caption:")
                st.write(result["caption"])
                st.subheader("Suggested Hashtags:")
                st.write(", ".join(result["hashtags"]))
        else:
            st.warning("Please enter an image URL.")

if __name__ == "__main__":
    main()