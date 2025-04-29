import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
import time
from datetime import datetime


class FoodComponentDataset(Dataset):
    """Dataset for training food component weight prediction model."""

    def __init__(self, dataset_index, image_dir, processed_dir, component_type=None, transform_mode='train'):
        """
        Args:
            dataset_index: Path to dataset index JSON file or loaded dataset
            image_dir: Directory containing original food images
            processed_dir: Directory containing processed masks
            component_type: If specified, only load this component type (e.g., 'egg')
            transform_mode: 'train' for training augmentations, 'val' for validation
        """
        if isinstance(dataset_index, str):
            # Load dataset index from file
            with open(dataset_index, 'r') as f:
                self.dataset = json.load(f)['samples']
        else:
            # Use provided dataset
            self.dataset = dataset_index

        self.image_dir = image_dir
        self.processed_dir = processed_dir
        self.component_type = component_type
        self.transform_mode = transform_mode

        # Define separate transforms for images and masks based on mode
        if transform_mode == 'train':
            # More aggressive augmentation for training
            self.image_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize larger for random crop
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])
        else:
            # Simple transformation for validation
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        # Create samples list
        self.samples = []
        for item in self.dataset:
            image_id = item['image_id']
            image_path = os.path.join(image_dir, item['image_path'])

            if not os.path.exists(image_path):
                # Try to find the image with different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = os.path.join(image_dir, f"{image_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break

            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {image_id}")
                continue

            # Handle either all components or just the specified one
            if component_type:
                if component_type in item['components']:
                    comp_data = item['components'][component_type]
                    mask_path = os.path.join(processed_dir, comp_data['mask_path'])

                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image_id': image_id,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'component': component_type,
                            'weight': comp_data['weight_grams'],
                            'features': {
                                'area': comp_data.get('area_pixels', 0),
                                'width': comp_data.get('width_pixels', 0),
                                'height': comp_data.get('height_pixels', 0),
                                'aspect_ratio': comp_data.get('aspect_ratio', 1.0),
                                'confidence': comp_data.get('confidence', 1.0)
                            }
                        })
            else:
                # Add all components
                for comp_name, comp_data in item['components'].items():
                    mask_path = os.path.join(processed_dir, comp_data['mask_path'])

                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image_id': image_id,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'component': comp_name,
                            'weight': comp_data['weight_grams'],
                            'features': {
                                'area': comp_data.get('area_pixels', 0),
                                'width': comp_data.get('width_pixels', 0),
                                'height': comp_data.get('height_pixels', 0),
                                'aspect_ratio': comp_data.get('aspect_ratio', 1.0),
                                'confidence': comp_data.get('confidence', 1.0)
                            }
                        })

        print(f"Loaded {len(self.samples)} samples for {'all components' if not component_type else component_type}")

        # Compute component type embeddings
        self.component_types = sorted(list(set(sample['component'] for sample in self.samples)))
        self.component_to_idx = {comp: idx for idx, comp in enumerate(self.component_types)}

        # Calculate normalization factors for manual features
        self.calculate_feature_stats()

    def calculate_feature_stats(self):
        """Calculate mean and std for manual features to improve normalization."""
        areas = [s['features']['area'] for s in self.samples if s['features']['area'] > 0]
        widths = [s['features']['width'] for s in self.samples if s['features']['width'] > 0]
        heights = [s['features']['height'] for s in self.samples if s['features']['height'] > 0]

        # Calculate statistics for feature normalization
        self.area_mean = np.mean(areas) if areas else 50000
        self.area_std = np.std(areas) if areas else 10000
        self.width_mean = np.mean(widths) if widths else 500
        self.width_std = np.std(widths) if widths else 100
        self.height_mean = np.mean(heights) if heights else 500
        self.height_std = np.std(heights) if heights else 100

        print(f"Feature statistics - Area: mean={self.area_mean:.1f}, std={self.area_std:.1f}")
        print(f"Feature statistics - Width: mean={self.width_mean:.1f}, std={self.width_std:.1f}")
        print(f"Feature statistics - Height: mean={self.height_mean:.1f}, std={self.height_std:.1f}")

    def __len__(self):
        return len(self.samples)

    def normalize_features(self, features):
        """Normalize features using z-score normalization with dataset statistics."""
        normalized = features.clone()

        # Z-score normalization (if area > 0)
        if features[0] > 0:
            normalized[0] = (features[0] - self.area_mean) / self.area_std
            normalized[1] = (features[1] - self.width_mean) / self.width_std
            normalized[2] = (features[2] - self.height_mean) / self.height_std

        return normalized

    def synchronized_transforms(self, image, mask):
        """Apply the same random transformations to both image and mask."""
        # Only for validation or when we want deterministic transforms
        if self.transform_mode != 'train':
            return self.image_transform(image), self.mask_transform(mask)

        # For training with synchronized transforms
        # First resize both to the same size
        image = transforms.Resize((256, 256))(image)
        mask = transforms.Resize((256, 256))(mask)

        # Generate random transformation parameters
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        flip = torch.rand(1) < 0.5
        angle = transforms.RandomRotation.get_params([-10, 10])

        # Apply the same transformations in sequence
        # 1. Random crop
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        # 2. Random horizontal flip
        if flip:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # 3. Random rotation
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)

        # 4. Image-specific transforms
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # 5. Mask-specific transforms
        mask = transforms.ToTensor()(mask)

        return image, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')

        # Apply transformations with synchronized random parameters
        image, mask = self.synchronized_transforms(image, mask)

        # Create one-hot encoding for component type
        component_idx = self.component_to_idx[sample['component']]
        component_onehot = torch.zeros(len(self.component_types))
        component_onehot[component_idx] = 1.0

        # Extract manual features
        manual_features = torch.tensor([
            sample['features']['area'],
            sample['features']['width'],
            sample['features']['height'],
            sample['features']['aspect_ratio'],
            sample['features']['confidence']
        ], dtype=torch.float32)

        # Normalize manual features using dataset statistics
        manual_features = self.normalize_features(manual_features)

        return {
            'image': image,
            'mask': mask,
            'component_type': component_onehot,
            'manual_features': manual_features,
            'weight': torch.tensor(sample['weight'], dtype=torch.float32),
            'component_name': sample['component']  # Add component name for analysis
        }


class FoodWeightCNN(nn.Module):
    """CNN model for predicting food component weights."""

    def __init__(self, num_components, use_manual_features=True, dropout_rate=0.3):
        """
        Args:
            num_components: Number of different food component types
            use_manual_features: Whether to use hand-crafted features
            dropout_rate: Dropout rate for regularization
        """
        super(FoodWeightCNN, self).__init__()

        # Image feature extraction with batch normalization
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # Mask feature extraction with batch normalization
        self.mask_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # Global average pooling to reduce feature maps
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Component type embedding
        self.component_embedding = nn.Sequential(
            nn.Linear(num_components, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2)  # Less dropout for this smaller network
        )

        # Number of features after pooling
        img_features = 64
        mask_features = 32
        comp_features = 16
        manual_features = 5 if use_manual_features else 0

        total_features = img_features + mask_features + comp_features + manual_features

        # Regression head with batch normalization
        self.regressor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1)
        )

        self.use_manual_features = use_manual_features

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize model weights for better convergence."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, image, mask, component_type, manual_features=None):
        # Extract image features
        img_feats = self.image_features(image)
        img_feats = self.gap(img_feats).squeeze(-1).squeeze(-1)

        # Extract mask features
        mask_feats = self.mask_features(mask)
        mask_feats = self.gap(mask_feats).squeeze(-1).squeeze(-1)

        # Component type embedding
        comp_feats = self.component_embedding(component_type)

        # Concatenate features
        if self.use_manual_features and manual_features is not None:
            combined = torch.cat([img_feats, mask_feats, comp_feats, manual_features], dim=1)
        else:
            combined = torch.cat([img_feats, mask_feats, comp_feats], dim=1)

        # Predict weight
        weight = self.regressor(combined)
        return weight.squeeze()


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0, verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, train_loader, criterion, optimizer, device, clip_value=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        component_types = batch['component_type'].to(device)
        manual_features = batch['manual_features'].to(device)
        weights = batch['weight'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, masks, component_types, manual_features)
        loss = criterion(outputs, weights)

        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Update progress bar with current loss
        progress_bar.set_postfix({'batch_loss': loss.item()})

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []
    component_names = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            component_types = batch['component_type'].to(device)
            manual_features = batch['manual_features'].to(device)
            weights = batch['weight'].to(device)

            # Forward pass
            outputs = model(images, masks, component_types, manual_features)
            loss = criterion(outputs, weights)

            total_loss += loss.item() * images.size(0)

            # Store predictions and ground truths for metrics
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(weights.cpu().numpy())
            component_names.extend(batch['component_name'])

    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - ground_truths))

    # Mean Absolute Percentage Error (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mapes = np.abs((predictions - ground_truths) / ground_truths) * 100
    mape = np.mean(mapes[~np.isnan(mapes) & ~np.isinf(mapes)])

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))

    return total_loss / len(val_loader.dataset), mae, mape, rmse, predictions, ground_truths, component_names


def plot_training_results(history, output_dir):
    """Plot training and validation loss with improved visualization."""
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Set plot style for better visualization
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot validation metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_mae'], 'g-', linewidth=2, label='MAE (grams)')
    plt.plot(epochs, history['val_rmse'], 'm-', linewidth=2, label='RMSE (grams)')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error (grams)', fontsize=12)
    plt.title('Validation Metrics', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    plt.close()

    # Plot prediction vs. ground truth scatter plot for last epoch
    plt.figure(figsize=(10, 8))
    plt.scatter(history['ground_truths'][-1], history['predictions'][-1],
                alpha=0.6, c='blue', edgecolors='k', s=60)

    # Add perfect prediction line
    min_val = min(min(history['ground_truths'][-1]), min(history['predictions'][-1]))
    max_val = max(max(history['ground_truths'][-1]), max(history['predictions'][-1]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    # Calculate R² for the plot
    correlation_matrix = np.corrcoef(history['ground_truths'][-1], history['predictions'][-1])
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    plt.xlabel('Ground Truth (grams)', fontsize=12)
    plt.ylabel('Prediction (grams)', fontsize=12)
    plt.title(
        f'Prediction vs. Ground Truth\nMAE: {history["val_mae"][-1]:.2f}g, MAPE: {history["val_mape"][-1]:.2f}%, R²: {r_squared:.3f}',
        fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_vs_truth.png'), dpi=300)
    plt.close()

    # Plot component-specific results if available
    if 'component_names' in history and 'component_predictions' in history:
        component_names = history['component_names']
        component_predictions = history['component_predictions'][-1]
        component_ground_truths = history['component_ground_truths'][-1]

        # Calculate how many subplot rows and columns we need
        n_components = len(component_names)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division

        plt.figure(figsize=(15, 5 * n_rows))
        for i, component in enumerate(component_names):
            plt.subplot(n_rows, n_cols, i + 1)
            pred = component_predictions[component]
            truth = component_ground_truths[component]

            if len(pred) > 0:
                plt.scatter(truth, pred, alpha=0.6, c='blue', edgecolors='k', s=50)

                # Add perfect prediction line
                min_val = min(min(truth), min(pred))
                max_val = max(max(truth), max(pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

                # Add component specific metrics
                mae = np.mean(np.abs(np.array(pred) - np.array(truth)))

                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_values = np.abs((np.array(pred) - np.array(truth)) / np.array(truth)) * 100
                mape = np.mean(mape_values[~np.isnan(mape_values) & ~np.isinf(mape_values)])

                # Calculate R² for component
                if len(truth) > 1:  # Need at least 2 points for correlation
                    r_squared_comp = np.corrcoef(truth, pred)[0, 1] ** 2
                    plt.title(f'{component.capitalize()}\nMAE: {mae:.2f}g, MAPE: {mape:.2f}%, R²: {r_squared_comp:.3f}')
                else:
                    plt.title(f'{component.capitalize()}\nMAE: {mae:.2f}g, MAPE: {mape:.2f}%')

                plt.xlabel('Ground Truth (g)')
                plt.ylabel('Prediction (g)')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'prediction_by_component.png'), dpi=300)
        plt.close()


def export_model(model, num_components, output_dir):
    """Export model to TorchScript for mobile deployment."""
    model.eval()

    # Create example inputs
    example_image = torch.randn(1, 3, 224, 224)
    example_mask = torch.randn(1, 1, 224, 224)
    example_component = torch.zeros(1, num_components)
    example_component[0, 0] = 1.0  # One-hot for first component
    example_manual = torch.randn(1, 5)

    # Trace model
    traced_model = torch.jit.trace(
        model,
        (example_image, example_mask, example_component, example_manual)
    )

    # Save model
    model_path = os.path.join(output_dir, 'weight_model_mobile.pt')
    traced_model.save(model_path)
    print(f"Exported TorchScript model to {model_path}")

    # Also save component names for reference
    component_info = {
        'component_names': model.component_types if hasattr(model, 'component_types') else [],
        'input_shape': {
            'image': [1, 3, 224, 224],
            'mask': [1, 1, 224, 224],
            'component_type': [1, num_components],
            'manual_features': [1, 5]
        },
        'model_version': 'v2',
        'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(component_info, f, indent=2)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def generate_history_and_plots(model, dataset, val_dataset, output_dir, device):
    """Generate training history and plots without training."""
    print("Generating history and plots for existing model...")

    # Set up criterion for validation
    criterion = nn.MSELoss()

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Validate model
    val_loss, val_mae, val_mape, val_rmse, predictions, ground_truths, component_names = validate(
        model, val_loader, criterion, device
    )

    # Print metrics
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val MAE: {val_mae:.2f}g, Val MAPE: {val_mape:.2f}%, Val RMSE: {val_rmse:.2f}g")

    # Get component-specific results
    component_pred_dict = {}
    component_truth_dict = {}

    for pred, truth, comp_name in zip(predictions, ground_truths, component_names):
        if comp_name not in component_pred_dict:
            component_pred_dict[comp_name] = []
            component_truth_dict[comp_name] = []

        component_pred_dict[comp_name].append(float(pred))
        component_truth_dict[comp_name].append(float(truth))

    # Create history dictionary with single entry
    history = {
        'train_loss': [0.0],  # Placeholder since we don't have training data
        'val_loss': [float(val_loss)],
        'val_mae': [float(val_mae)],
        'val_mape': [float(val_mape)],
        'val_rmse': [float(val_rmse)],
        'predictions': [[float(p) for p in predictions.tolist()]],
        'ground_truths': [[float(g) for g in ground_truths.tolist()]],
        'component_predictions': [component_pred_dict],
        'component_ground_truths': [component_truth_dict],
        'component_names': dataset.component_types
    }

    # Save history
    serializable_history = convert_numpy_types(history)
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2)

    # Plot results
    plot_training_results(history, output_dir)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train food component weight prediction model (V2)')
    parser.add_argument('--dataset_index', type=str, default='processed_dataset/dataset_index.json',
                        help='Path to dataset index JSON file')
    parser.add_argument('--image_dir', type=str, default='.',
                        help='Directory containing original images')
    parser.add_argument('--processed_dir', type=str, default='processed_dataset',
                        help='Directory containing processed masks')
    parser.add_argument('--output_dir', type=str, default='model_outputV2',
                        help='Directory to save model and results')
    parser.add_argument('--component_type', type=str, default=None,
                        help='Train for specific component type (default: all components)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_manual_features', action='store_true',
                        help='Use manual features (area, width, etc.)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training if model exists and just generate plots')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda or cpu, default: auto-detect)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directory with subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save arguments for reference
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("Loading dataset...")
    full_dataset = FoodComponentDataset(
        args.dataset_index,
        args.image_dir,
        args.processed_dir,
        component_type=args.component_type,
        transform_mode='train'  # Initially set to train mode
    )

    # Split into train/validation sets
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=args.seed,
        shuffle=True
    )

    # Create train dataset with augmentations
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    # Create validation dataset with separate instance using validation transforms
    val_full_dataset = FoodComponentDataset(
        args.dataset_index,
        args.image_dir,
        args.processed_dir,
        component_type=args.component_type,
        transform_mode='val'  # Validation mode with no augmentations
    )
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # Initialize model
    num_components = len(full_dataset.component_types)
    model = FoodWeightCNN(
        num_components=num_components,
        use_manual_features=args.use_manual_features,
        dropout_rate=args.dropout
    )

    # Save component types to model for reference
    model.component_types = full_dataset.component_types
    model.component_to_idx = full_dataset.component_to_idx

    # Check if we should skip training
    model_path = os.path.join(args.output_dir, 'checkpoints', 'best_weight_model.pth')
    if os.path.exists(model_path) and args.skip_training:
        print(f"Found existing model at {model_path}, skipping training.")

        # Load the existing model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Generate history and plots
        generate_history_and_plots(model, full_dataset, val_dataset, args.output_dir, device)

        print("Analysis completed!")
        return

    # Continue with training if not skipping
    model.to(device)

    # Print model summary
    print(f"Model architecture:")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create data loaders for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Loss function and optimizer with weight decay for L2 regularization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=os.path.join(args.output_dir, 'checkpoints', 'best_weight_model.pth')
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mape': [],
        'val_rmse': [],
        'predictions': [],
        'ground_truths': [],
        'component_predictions': [],
        'component_ground_truths': [],
        'component_names': full_dataset.component_types,
        'learning_rates': []
    }

    # Start time for training
    start_time = time.time()

    # Training loop
    print(f"Starting training for up to {args.epochs} epochs...")

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.6f})")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_mae, val_mape, val_rmse, predictions, ground_truths, component_names = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step()

        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        # Print metrics
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Val MAE: {val_mae:.2f}g, Val MAPE: {val_mape:.2f}%, Val RMSE: {val_rmse:.2f}g")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.1f} seconds")

        # Save history - convert numpy values to Python native types
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_mape'].append(float(val_mape))
        history['val_rmse'].append(float(val_rmse))

        # Convert numpy arrays to native Python lists with native Python floats
        history['predictions'].append([float(p) for p in predictions.tolist()])
        history['ground_truths'].append([float(g) for g in ground_truths.tolist()])

        # Get per-component predictions
        component_predictions = {comp: [] for comp in full_dataset.component_types}
        component_ground_truths = {comp: [] for comp in full_dataset.component_types}

        # Group predictions by component
        for pred, truth, comp_name in zip(predictions, ground_truths, component_names):
            component_predictions[comp_name].append(float(pred))
            component_ground_truths[comp_name].append(float(truth))

        history['component_predictions'].append(component_predictions)
        history['component_ground_truths'].append(component_ground_truths)

        # Save intermediate history and plots every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Convert any remaining numpy types before saving
            serializable_history = convert_numpy_types(history)

            # Save training history
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(serializable_history, f, indent=2)

            # Plot intermediate results
            plot_training_results(history, args.output_dir)

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

    # End time for training
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)!")

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_weight_model.pth')))

    # Final evaluation with the best model
    print("Performing final evaluation with best model...")
    val_loss, val_mae, val_mape, val_rmse, predictions, ground_truths, component_names = validate(
        model, val_loader, criterion, device
    )

    print(f"Best model - Val Loss: {val_loss:.6f}")
    print(f"Best model - Val MAE: {val_mae:.2f}g, Val MAPE: {val_mape:.2f}%, Val RMSE: {val_rmse:.2f}g")

    # Save best model statistics
    best_model_stats = {
        'val_loss': float(val_loss),
        'val_mae': float(val_mae),
        'val_mape': float(val_mape),
        'val_rmse': float(val_rmse),
        'training_time_seconds': training_time,
        'epochs_trained': len(history['train_loss']),
        'early_stopped': len(history['train_loss']) < args.epochs
    }

    with open(os.path.join(args.output_dir, 'best_model_stats.json'), 'w') as f:
        json.dump(best_model_stats, f, indent=2)

    # Export model for mobile
    export_model(model, num_components, args.output_dir)

    # Save final history and plots
    serializable_history = convert_numpy_types(history)
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2)

    plot_training_results(history, args.output_dir)

    print("Training and analysis completed!")


if __name__ == "__main__":
    main()