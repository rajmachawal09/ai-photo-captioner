import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForImageClassification, VisionEncoderDecoderModel

def load_models():
    try:
        print("Loading image classification model...")
        image_classifier = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
        print("Loading image captioning model...")
        caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        caption_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        return image_classifier, image_processor, caption_model, caption_processor, caption_tokenizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error fetching image: {str(e)}")
        return None

def analyze_image(image_url, image_classifier, image_processor):
    image = get_image_from_url(image_url)
    if image is None:
        return None
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = image_classifier(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return image_classifier.config.id2label[predicted_class_idx]

def generate_caption(image_url, caption_model, caption_processor, caption_tokenizer):
    image = get_image_from_url(image_url)
    if image is None:
        return None
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
    
    # Generate caption with attention mask
    max_length = 30
    output_ids = caption_model.generate(
        pixel_values, 
        max_length=max_length,
        num_beams=4,
        attention_mask=torch.ones(pixel_values.shape[0], max_length)
    )
    
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def generate_hashtags(image_content):
    return [f"#{word.lower()}" for word in image_content.split() if len(word) > 3]

def process_image(image_url, models):
    image_classifier, image_processor, caption_model, caption_processor, caption_tokenizer = models
    
    try:
        image_content = analyze_image(image_url, image_classifier, image_processor)
        if image_content is None:
            return {"error": "Failed to analyze image content"}
        
        caption = generate_caption(image_url, caption_model, caption_processor, caption_tokenizer)
        if caption is None:
            return {"error": "Failed to generate caption"}
        
        hashtags = generate_hashtags(image_content)
        
        return {
            "content": image_content,
            "caption": caption,
            "hashtags": hashtags
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def main():
    print("Loading models... This may take a few minutes.")
    models = load_models()
    if any(model is None for model in models):
        print("Failed to load models. Exiting.")
        return

    # image_url = input("Please enter the URL of the image you want to analyze: ")
    image_url = "https://hips.hearstapps.com/hmg-prod/images/champagne-beach-espiritu-santo-island-vanuatu-royalty-free-image-1655672510.jpg"
    
    result = process_image(image_url, models)
    
    if "error" in result:
        print(result["error"])
    else:
        print(f"Image content: {result['content']}")
        print(f"Generated caption: {result['caption']}")
        print(f"Suggested hashtags: {', '.join(result['hashtags'])}")

if __name__ == "__main__":
    main()