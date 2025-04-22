import os
import json
import uuid
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from train_weight_model import FoodWeightCNN

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'snapsilog-v1-secret-key')  # Use env var in production
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', 'static/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Roboflow API settings
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', "FSX72icxHTJs0335DLkL")
ROBOFLOW_MODEL = os.environ.get('ROBOFLOW_MODEL', "snapsilog/3")
API_URL = f"https://outline.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"

# Model and nutrition data paths
MODEL_PATH = os.environ.get('MODEL_PATH', 'model_output/best_weight_model.pth')
MODEL_INFO_PATH = os.environ.get('MODEL_INFO_PATH', 'model_output/model_info.json')
NUTRITION_DATA_PATH = os.environ.get('NUTRITION_DATA_PATH', 'nutritional_database.json')

# Global variables for model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
nutrition_data = None

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_nutritional_data(file_path):
    """Load nutritional data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_model(model_path, model_info_path, device):
    """Load the trained weight prediction model."""
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    # Get the number of food components
    num_components = len(model_info['component_names'])

    print(f"Creating model with {num_components} components")

    # The issue is likely that you trained with a different number of components
    # than what's in your current info file. Let's hardcode the model architecture
    # to match exactly what was used during training.

    model = FoodWeightCNN(num_components=7, use_manual_features=True)

    # Now load the weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    except RuntimeError as e:
        # If there's a mismatch, print details and try to fix it
        print(f"Error loading model: {str(e)}")

        # This is a fallback approach - we'll rebuild the model with a hardcoded architecture
        # that matches what was used during training
        print("Trying alternate model configuration...")

        # Image feature extraction
        img_features = 64
        mask_features = 32
        comp_features = 16
        manual_features = 0  # Try without manual features

        total_features = img_features + mask_features + comp_features + manual_features

        # Recreate the regressor with the expected size from training
        model.regressor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # Set use_manual_features to False to match the architecture
        model.use_manual_features = False

        # Try loading again
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully with modified architecture")

    model.to(device)
    model.eval()

    # Store component types in model
    model.component_types = model_info['component_names']
    model.component_to_idx = {comp: idx for idx, comp in enumerate(model_info['component_names'])}

    print(f"Model is using manual features: {model.use_manual_features}")

    return model


def upload_image_to_roboflow(image_path):
    """Upload image to Roboflow API with resizing if needed."""
    try:
        # Resize the image to fit within Roboflow limits
        resized_path = resize_image_for_api(image_path)

        # Read image
        with open(resized_path, "rb") as f:
            image_data = f.read()

        print(f"Image size to be uploaded: {len(image_data)} bytes")

        # Convert image to base64
        encoded_image = base64.b64encode(image_data).decode("utf-8")

        # Prepare the API URL - use the detect endpoint instead of outline
        api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"

        # Make the API request with proper content type header
        response = requests.post(
            api_url,
            data=encoded_image,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        print(f"Roboflow API status code: {response.status_code}")

        if response.status_code != 200:
            print(f"Error from Roboflow API: {response.text}")
            return None

        return response.json()

    except Exception as e:
        print(f"Error sending image to Roboflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_mask_data(prediction, img_width, img_height):
    """Extract and process mask data from API response."""
    if 'mask' in prediction:
        # Base64 encoded mask
        try:
            mask_data = base64.b64decode(prediction['mask'])
            mask_array = np.frombuffer(mask_data, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            mask = cv2.resize(mask, (img_width, img_height))
            return mask
        except Exception as e:
            print(f"Error processing mask: {str(e)}")
            return None
    else:
        print("No mask data available")
        return None


def resize_image_for_api(image_path, max_size=(640, 640)):
    """Resize an image to be within the specified dimensions while maintaining aspect ratio."""
    try:
        # Open the image
        img = Image.open(image_path)

        # Get original dimensions
        orig_width, orig_height = img.size
        print(f"Original image size: {orig_width}x{orig_height} pixels")

        # Check if resizing is needed
        if orig_width <= max_size[0] and orig_height <= max_size[1]:
            print("Image already within size limits, no resizing needed")
            return image_path

        # Calculate the scaling factor
        width_ratio = max_size[0] / orig_width
        height_ratio = max_size[1] / orig_height
        ratio = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image
        filename = os.path.basename(image_path)
        base_name, extension = os.path.splitext(filename)
        resized_path = os.path.join(os.path.dirname(image_path), f"{base_name}_resized{extension}")
        resized_img.save(resized_path)

        print(f"Resized image to {new_width}x{new_height} pixels, saved as {resized_path}")
        return resized_path

    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return image_path


def detect_food(image_path):
    """Detect food components using Roboflow API."""
    try:
        # Resize the image if necessary
        resized_path = resize_image_for_api(image_path)

        # Get image dimensions from the resized image
        img = Image.open(resized_path)
        img_width, img_height = img.size

        # Get Roboflow API response directly
        result = upload_image_to_roboflow(resized_path)

        if not result:
            print("Failed to get response from Roboflow API")
            return None

        print(f"Roboflow API response: {json.dumps(result, indent=2)}")

        # Skip if no predictions
        if "predictions" not in result or len(result["predictions"]) == 0:
            print("No food detected in the image")
            return None

        # Process each component
        components = {}
        silog_type = None

        # Read image for visualization (use original image for visualization)
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]  # Get actual dimensions from the image
        print(f"Original image dimensions from cv2: {orig_width}x{orig_height}")

        # Create blank overlay for visualization with correct dimensions
        overlay = np.zeros_like(image)

        # Colors for different components (BGR format)
        colors = {
            "tapa": (0, 0, 255),  # Red
            "hotdog": (0, 165, 255),  # Orange
            "ham": (0, 255, 255),  # Yellow
            "spam": (255, 0, 255),  # Magenta
            "porkchop": (255, 0, 0),  # Blue
            "egg": (0, 255, 0),  # Green
            "rice": (255, 255, 0)  # Cyan
        }

        print(f"Processing {len(result['predictions'])} predictions")

        for prediction in result["predictions"]:
            class_name = prediction["class"].lower()
            print(f"Processing class: {class_name}, confidence: {prediction['confidence']}")

            # Check if this is a silog type
            if class_name.endswith("silog"):
                silog_type = class_name
                continue

            # Process mask for this component
            mask = None

            # Scale factors to map predictions back to original image
            # Use the actual dimensions from cv2.imread
            scale_x = orig_width / img_width
            scale_y = orig_height / img_height

            print(f"Scale factors: x={scale_x}, y={scale_y}")

            # Check if points are available (polygon segmentation)
            if "points" in prediction and prediction["points"]:
                print(f"Using polygon points for {class_name}")

                # Create a list of points as (x, y) tuples
                polygon_points = []

                # Handle different formats of points in the API response
                if isinstance(prediction["points"], list):
                    if isinstance(prediction["points"][0], dict):
                        # Format: [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]
                        for point in prediction["points"]:
                            # Scale coordinates to original image size
                            x = int(point["x"] * scale_x)
                            y = int(point["y"] * scale_y)
                            polygon_points.append((x, y))
                    else:
                        # Format: [x1, y1, x2, y2, ...]
                        for i in range(0, len(prediction["points"]), 2):
                            x = int(prediction["points"][i] * scale_x)
                            y = int(prediction["points"][i+1] * scale_y)
                            polygon_points.append((x, y))

                # Convert to numpy array for OpenCV
                points_array = np.array(polygon_points)

                # Create an empty mask for the original image
                mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

                # Draw filled polygon
                if len(points_array) > 2:  # Need at least 3 points for a polygon
                    cv2.fillPoly(mask, [points_array], 255)
                else:
                    print(f"Not enough points for polygon for {class_name}")
                    continue

            elif "mask" in prediction:
                print(f"Using provided mask for {class_name}")
                # Base64 encoded mask
                try:
                    mask_data = base64.b64decode(prediction["mask"])
                    mask_array = np.frombuffer(mask_data, dtype=np.uint8)
                    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Resize mask to match original image dimensions from cv2.imread
                        mask = cv2.resize(mask, (orig_width, orig_height))
                except Exception as e:
                    print(f"Error processing mask: {str(e)}")
            else:
                print(f"No mask or points data for {class_name}, using bounding box")
                # If no mask or points, use bounding box
                x = int(prediction["x"] - prediction["width"] / 2)
                y = int(prediction["y"] - prediction["height"] / 2)
                width = int(prediction["width"])
                height = int(prediction["height"])

                # Scale bounding box to original image
                x = int(x * scale_x)
                y = int(y * scale_y)
                width = int(width * scale_x)
                height = int(height * scale_y)

                # Ensure coordinates are within image bounds
                x = max(0, min(x, orig_width-1))
                y = max(0, min(y, orig_height-1))
                width = min(width, orig_width - x)
                height = min(height, orig_height - y)

                # Create mask from bounding box
                mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                mask[y:y+height, x:x+width] = 255

            if mask is None:
                print(f"Failed to create mask for {class_name}")
                continue

            print(f"Mask shape: {mask.shape}, Overlay shape: {overlay.shape}")

            # Verify mask and overlay dimensions match
            if mask.shape[:2] != overlay.shape[:2]:
                print(f"Error: Mask shape {mask.shape[:2]} does not match overlay shape {overlay.shape[:2]}")
                # Resize mask to match overlay
                mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]))
                print(f"Resized mask to {mask.shape}")

            # Calculate features
            area = np.sum(mask > 0)

            if area > 0:
                # Get non-zero pixel coordinates
                y_indices, x_indices = np.where(mask > 0)

                # Calculate bounding box
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                width = max_x - min_x + 1
                height = max_y - min_y + 1

                # Calculate aspect ratio
                aspect_ratio = width / height if height > 0 else 0

                # Store component data
                components[class_name] = {
                    'mask': mask,
                    'area_pixels': int(area),
                    'width_pixels': int(width),
                    'height_pixels': int(height),
                    'aspect_ratio': float(aspect_ratio),
                    'confidence': float(prediction['confidence']),
                    'bbox': [min_x, min_y, max_x, max_y]
                }

                # Add color to overlay for visualization
                color = colors.get(class_name, (255, 255, 255))  # Default to white
                try:
                    overlay[mask > 0] = color
                except IndexError as e:
                    print(f"Error applying mask to overlay: {str(e)}")
                    continue

                # Save the individual mask as image for debugging
                debug_mask_path = os.path.join('static/results', f'debug_{class_name}_mask.png')
                cv2.imwrite(debug_mask_path, mask)
                print(f"Saved debug mask for {class_name} to {debug_mask_path}")

        # Check if we found any components
        if not components:
            print("No food components were successfully processed")
            return None

        # Create visualization
        if components:
            # Blend with original image
            alpha = 0.5
            output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Save visualization
            filename = os.path.basename(image_path)
            base_filename = os.path.splitext(filename)[0]
            viz_path = os.path.join('static/results', f'{base_filename}_detection.jpg')
            cv2.imwrite(viz_path, output)
            print(f"Saved detection visualization to {viz_path}")

            return {
                'silog_type': silog_type,
                'components': components,
                'image_size': (orig_width, orig_height),
                'visualization': os.path.relpath(viz_path, 'static')
            }
        else:
            print("No components detected")
            return None

    except Exception as e:
        print(f"Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_weights(detection_result, model, device, image_path):
    """Predict component weights using the trained model."""
    if not detection_result or 'components' not in detection_result:
        return None

    # Load and preprocess image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Separate transform for masks (no normalization)
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # Process each component
    weight_predictions = {}
    total_weight = 0.0

    for component_name, component_data in detection_result['components'].items():
        try:
            # Skip if component not in model's component types
            if component_name not in model.component_types:
                print(f"Warning: Component {component_name} not in model's component types, skipping.")
                print(f"Available components: {model.component_types}")
                continue

            # Prepare mask tensor - use mask_transform instead of the same transform as image
            mask = component_data['mask']
            mask_pil = Image.fromarray(mask).convert('L')
            mask_tensor = mask_transform(mask_pil).unsqueeze(0).to(device)

            # Prepare component one-hot tensor
            component_idx = model.component_to_idx[component_name]
            component_tensor = torch.zeros(1, len(model.component_types)).to(device)
            component_tensor[0, component_idx] = 1.0

            # Prepare manual features tensor
            img_width, img_height = detection_result['image_size']
            img_size = img_width * img_height

            manual_features = torch.tensor([
                component_data['area_pixels'] / img_size,
                component_data['width_pixels'] / img_width,
                component_data['height_pixels'] / img_height,
                component_data['aspect_ratio'],
                component_data['confidence']
            ], dtype=torch.float32).unsqueeze(0).to(device)

            # Predict weight - handle different model configurations
            with torch.no_grad():
                if model.use_manual_features:
                    weight = model(image_tensor, mask_tensor, component_tensor, manual_features).item()
                else:
                    weight = model(image_tensor, mask_tensor, component_tensor).item()

            # Store prediction
            weight_predictions[component_name] = weight
            total_weight += weight

            print(f"Predicted weight for {component_name}: {weight:.2f}g")

        except Exception as e:
            print(f"Error predicting weight for {component_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not weight_predictions:
        print("No weights could be predicted for any components")
        return None

    print(f"Total predicted weight: {total_weight:.2f}g")

    return {
        'component_weights': weight_predictions,
        'total_weight': total_weight
    }


def calculate_nutrition(weight_predictions, nutritional_data):
    """Calculate nutritional information based on component weights."""
    if not weight_predictions or 'component_weights' not in weight_predictions:
        return None

    nutrition_info = {
        'components': {},
        'total': {
            'weight_grams': weight_predictions['total_weight'],
            'calories': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbs': 0.0
        }
    }

    for component, weight in weight_predictions['component_weights'].items():
        # Skip if component not in nutritional data
        if component not in nutritional_data['nutritional_data']:
            print(f"Warning: Component {component} not in nutritional data, skipping.")
            continue

        # Get nutritional data for this component
        comp_nutrition = nutritional_data['nutritional_data'][component]

        # Calculate nutrition
        calories = weight * comp_nutrition['calories_per_gram']
        protein = weight * comp_nutrition['protein_per_gram']
        fat = weight * comp_nutrition['fat_per_gram']
        carbs = weight * comp_nutrition['carbs_per_gram']

        # Store component nutrition
        nutrition_info['components'][component] = {
            'weight_grams': weight,
            'calories': calories,
            'protein': protein,
            'fat': fat,
            'carbs': carbs
        }

        # Add to total
        nutrition_info['total']['calories'] += calories
        nutrition_info['total']['protein'] += protein
        nutrition_info['total']['fat'] += fat
        nutrition_info['total']['carbs'] += carbs

    return nutrition_info


def visualize_nutrition(image_path, detection_result, nutrition_info):
    """Create a visualization of the nutrition results."""
    if not detection_result or not nutrition_info:
        return None

    # Read image
    image = cv2.imread(image_path)

    # Create a copy for visualization
    visual = image.copy()

    # Add nutrition info overlay
    overlay = np.ones((image.shape[0] + 300, image.shape[1], 3), dtype=np.uint8) * 255
    overlay[0:image.shape[0], 0:image.shape[1]] = visual

    # Add title
    silog_type = detection_result['silog_type'] or "Unknown Silog"
    cv2.putText(overlay, f"{silog_type.upper()} - Nutrition Information",
                (20, image.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Add total nutrition info
    total = nutrition_info['total']
    cv2.putText(overlay, f"Total Weight: {total['weight_grams']:.1f}g",
                (20, image.shape[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay, f"Total Calories: {total['calories']:.1f} kcal",
                (20, image.shape[0] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay, f"Protein: {total['protein']:.1f}g | Fat: {total['fat']:.1f}g | Carbs: {total['carbs']:.1f}g",
                (20, image.shape[0] + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add component breakdown
    y_offset = image.shape[0] + 170
    cv2.putText(overlay, "Component Breakdown:",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_offset += 30

    # Colors for different components (BGR format)
    colors = {
        "tapa": (0, 0, 255),  # Red
        "hotdog": (0, 165, 255),  # Orange
        "ham": (0, 255, 255),  # Yellow
        "spam": (255, 0, 255),  # Magenta
        "porkchop": (255, 0, 0),  # Blue
        "egg": (0, 255, 0),  # Green
        "rice": (255, 255, 0)  # Cyan
    }

    for component, data in nutrition_info['components'].items():
        color = colors.get(component, (0, 0, 0))
        cv2.putText(overlay, f"{component.capitalize()}: {data['weight_grams']:.1f}g, {data['calories']:.1f} kcal",
                    (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    # Label components on the image
    for component, comp_data in detection_result['components'].items():
        if component in nutrition_info['components']:
            bbox = comp_data['bbox']
            color = colors.get(component, (0, 0, 0))
            cv2.rectangle(overlay[0:image.shape[0], 0:image.shape[1]],
                          (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            weight = nutrition_info['components'][component]['weight_grams']
            calories = nutrition_info['components'][component]['calories']
            cv2.putText(overlay[0:image.shape[0], 0:image.shape[1]],
                        f"{component}: {weight:.1f}g",
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save visualization
    filename = os.path.basename(image_path)
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f'{base_filename}_nutrition.jpg')
    cv2.imwrite(output_path, overlay)

    # Create chart visualization
    plt.figure(figsize=(10, 6))

    # Pie chart for calories
    plt.subplot(1, 2, 1)
    labels = list(nutrition_info['components'].keys())
    calorie_values = [nutrition_info['components'][comp]['calories'] for comp in labels]
    colors_rgb = [tuple(x / 255 for x in colors.get(comp, (0, 0, 0))[::-1]) for comp in labels]
    plt.pie(calorie_values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_rgb)
    plt.title('Calorie Distribution')

    # Stacked bar for macronutrients
    plt.subplot(1, 2, 2)
    components = list(nutrition_info['components'].keys())
    protein = [nutrition_info['components'][comp]['protein'] for comp in components]
    fat = [nutrition_info['components'][comp]['fat'] for comp in components]
    carbs = [nutrition_info['components'][comp]['carbs'] for comp in components]

    x = np.arange(len(components))
    width = 0.6

    plt.bar(x, protein, width, label='Protein', color='royalblue')
    plt.bar(x, fat, width, bottom=protein, label='Fat', color='salmon')
    plt.bar(x, carbs, width, bottom=np.array(protein) + np.array(fat), label='Carbs', color='khaki')

    plt.ylabel('Grams')
    plt.title('Macronutrient Composition')
    plt.xticks(x, components, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    chart_path = os.path.join(app.config['RESULTS_FOLDER'], f'{base_filename}_chart.png')
    plt.savefig(chart_path)
    plt.close()

    return {
        'nutrition_image': os.path.relpath(output_path, 'static'),
        'chart_image': os.path.relpath(chart_path, 'static')
    }


@app.route('/')
def index():
    """Simple health check route for API."""
    return jsonify({
        'status': 'ok',
        'message': 'Silog Nutrition API is running',
        'version': '1.0'
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing images."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        filename = unique_id + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Image saved to {filepath}")

        # Process the image
        detection_result = detect_food(filepath)
        if not detection_result:
            return jsonify({'error': 'No food components detected'}), 400

        weight_predictions = predict_weights(detection_result, model, device, filepath)
        if not weight_predictions:
            return jsonify({'error': 'Weight prediction failed'}), 400

        nutrition_info = calculate_nutrition(weight_predictions, nutrition_data)
        if not nutrition_info:
            return jsonify({'error': 'Nutrition calculation failed'}), 400

        visualization = visualize_nutrition(filepath, detection_result, nutrition_info)

        # Build server URL dynamically
        server_url = request.url_root
        if server_url.endswith('/'):
            server_url = server_url[:-1]

        # Prepare API response
        response = {
            'silog_type': detection_result.get('silog_type', 'Unknown Silog'),
            'total_weight': weight_predictions['total_weight'],
            'total_calories': nutrition_info['total']['calories'],
            'total_protein': nutrition_info['total']['protein'],
            'total_fat': nutrition_info['total']['fat'],
            'total_carbs': nutrition_info['total']['carbs'],
            'components': {},
            'images': {
                'original': f"{server_url}/static/uploads/{filename}",
                'detection': f"{server_url}/static/{detection_result.get('visualization')}" if detection_result.get('visualization') else None,
                'nutrition': f"{server_url}/static/{visualization.get('nutrition_image')}" if visualization else None,
                'chart': f"{server_url}/static/{visualization.get('chart_image')}" if visualization else None
            }
        }

        for component, data in nutrition_info['components'].items():
            response['components'][component] = {
                'weight': data['weight_grams'],
                'calories': data['calories'],
                'protein': data['protein'],
                'fat': data['fat'],
                'carbs': data['carbs']
            }

        return jsonify(response)

    return jsonify({'error': 'Invalid file type'}), 400


# Initialize app function - load models outside if __name__ block for cloud environments
def initialize_app():
    global model, nutrition_data
    print("Loading weight prediction model...")
    model = load_model(MODEL_PATH, MODEL_INFO_PATH, device)

    print("Loading nutritional database...")
    nutrition_data = load_nutritional_data(NUTRITION_DATA_PATH)


# Initialize on import
initialize_app()

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))

    # Run the app - debug should be False in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)