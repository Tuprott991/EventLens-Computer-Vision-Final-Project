import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from model_arch import EventLens

# from gcn_image_tagger_simplified import ImageGCNSimple

# Set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for ResNet models (fixed at 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes=23):
    """
    Load a trained ResNet backbone model
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of classes
        
    Returns:
        model: Loaded model
    """
    # Initialize model architecture - fixed for ResNet
    model = EventLens()
    
    
    model.to(device)
    model.eval()
    return model

def predict_tags(model, image, class_names, threshold=0.5):
    """
    Predict tags for an image
    
    Args:
        model: Trained model
        image: PIL Image
        class_names: List of class names
        threshold: Confidence threshold for predictions
        
    Returns:
        tags: Dictionary of class name -> confidence score
        top_tags: List of top tag names (above threshold)
    """
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Create tag dictionary
    tags = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # Get tags above threshold
    top_tags = [class_names[i] for i in range(len(class_names)) if probabilities[i] >= threshold]
    
    return tags, top_tags

def get_class_names(dataset="voc"):
    """
    Get class names for the specified dataset
    
    Args:
        dataset: Dataset name ('voc' or 'coco')
        
    Returns:
        class_names: List of class names
    """
    if dataset.lower() == "voc":
        return [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    elif dataset.lower() == "coco":
        # COCO dataset has 80 classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def tag_image_file(image_path, model_path, dataset="voc", threshold=0.5):
    """
    Tag an image file using the trained ResNet model
    
    Args:
        image_path: Path to image file
        model_path: Path to model weights
        dataset: Dataset name for class names ('voc' or 'coco')
        threshold: Confidence threshold
        
    Returns:
        image: Original image
        tags: Dictionary of class name -> confidence score
        top_tags: List of top tag names (above threshold)
    """
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, {}, []
    
    # Get class names
    class_names = get_class_names(dataset)
    
    # Load model
    model = load_model(
        model_path=model_path,
        num_classes=len(class_names)
    )
    
    # Predict tags
    tags, top_tags = predict_tags(model, image, class_names, threshold)
    
    return image, tags, top_tags

def display_results(image, tags, top_tags):
    """Display results in terminal"""
    print("\nPredicted Tags:")
    for tag in top_tags:
        print(f"- {tag}: {tags[tag]:.4f}")
    
    if not top_tags:
        print("No tags predicted above threshold")

def gradio_interface(image, model_path, dataset, threshold):
    """
    Gradio interface function for image tagging with ResNet
    
    Args:
        image: Input image
        model_path: Path to model
        dataset: Dataset for class names
        threshold: Confidence threshold
        
    Returns:
        result_image: Image with tags
        result_html: HTML with tag list
    """
    if image is None:
        return None, "Please upload an image"
    
    # Get class names
    try:
        class_names = get_class_names(dataset)
    except ValueError:
        return None, "Unknown dataset. Please use 'voc' or 'coco'."
    
    # Load model
    model = load_model(
        model_path=model_path,
        num_classes=len(class_names)
    )
    
    # Predict tags
    tags, top_tags = predict_tags(model, image, class_names, threshold)
    
    # Create result image with tags
    result_image = image.copy()
    
    # Generate HTML for results
    if not top_tags:
        result_html = "<div style='color: red'>No tags found above threshold.</div>"
    else:
        result_html = "<div style='text-align: left'><h3>Predicted Tags:</h3><ul>"
        for tag in sorted(top_tags):
            result_html += f"<li><b>{tag}</b>: {tags[tag]:.4f}</li>"
        result_html += "</ul></div>"
    
    return result_image, result_html

def launch_web_interface(model_path, dataset, threshold):
    """Launch the web interface for ResNet model"""
    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda img: gradio_interface(img, model_path, dataset, threshold),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.HTML(label="Predicted Tags")
        ],
        title="GCN Image Tagger (ResNet Backbone)",
        description=(
            "Upload an image to get predicted tags using a trained GCN model with ResNet backbone. "
            f"Using model: {os.path.basename(model_path)}, "
            f"dataset: {dataset}, threshold: {threshold}"
        ),
    )
    
    # Launch the interface
    iface.launch(share=True)

def main():
    parser = argparse.ArgumentParser(description='GCN Image Tagger with ResNet Backbone')
    parser.add_argument('--image', type=str, help='Path to an image file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco'], 
                        help='Dataset for class names')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for predictions')
    parser.add_argument('--web', action='store_true', help='Launch web interface')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Launch web interface if requested or if no image provided
    if args.web or args.image is None:
        try:
            import gradio as gr
            print("Launching web interface for ResNet model...")
            launch_web_interface(
                model_path=args.model,
                dataset=args.dataset,
                threshold=args.threshold
            )
        except ImportError:
            print("Error: Gradio not installed. Install it with 'pip install gradio'")
            print("Or provide an image path using --image")
    else:
        # Process single image
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        # Tag the image
        image, tags, top_tags = tag_image_file(
            image_path=args.image,
            model_path=args.model,
            dataset=args.dataset,
            threshold=args.threshold
        )
        
        # Display results
        if image is not None:
            display_results(image, tags, top_tags)

if __name__ == "__main__":
    main() 