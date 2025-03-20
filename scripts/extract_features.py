import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from config import get_config
from data.dataset import DDMDataset  # Import DDMDataset to access dataset path
from data.clip import TimmWrapper  # Assuming TimmWrapper is used for DINOv2

def extract_features(config_path="config.py", output_dir="/home/alex/workspace/Decentralized-Diffusion-Models/features"):
    """
    Precomputes DINOv2 features for the training dataset and saves them to disk.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Directory to save the extracted features.
    """
    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DINOv2 model
    print(f"Loading DINOv2 model: {config.clip_model}")
    dinov2_model = TimmWrapper(
        model_name=config.clip_model,
        pretrained=True,
        device=device,
         NormLayer=torch.nn.LayerNorm # Corrected: Pass the class itself, not an instance
    )
    dinov2_model.eval() # Set to eval mode

    # Initialize dataset to get image paths and dimensions
    print(f"Loading dataset from: {config.dataset_path}")
    dataset = DDMDataset(config, split='train', hf_split='train') # Using hf_split to avoid errors in dataset init

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    feature_output_path = os.path.join(output_dir, "train_features.pt")
    dims_output_path = os.path.join(output_dir, "dim_cache.pt")

    all_features = []
    all_dims = []

    print("Extracting features...")
    for image_path in tqdm(dataset.image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert('RGB')
            dims = image.size # (width, height)
            all_dims.append(dims)

            # Preprocess image for DINOv2 (adjust as needed for your DINOv2 wrapper)
            preprocess = transforms.Compose([
                transforms.Resize(256), # Example size, adjust if needed
                transforms.CenterCrop(224), # Example crop, adjust if needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
            ])
            input_tensor = preprocess(image).unsqueeze(0).to(device) # Add batch dimension

            # Extract features
            with torch.no_grad():
                features = dinov2_model(input_tensor) # Assuming your wrapper returns features directly

            all_features.append(features.cpu()) # Move features to CPU and store

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue # Skip to the next image in case of error

    # Save features and dimensions
    print(f"Saving features to: {feature_output_path}")
    all_features_tensor = torch.cat(all_features, dim=0) # Concatenate list of tensors to a single tensor
    torch.save(all_features_tensor, feature_output_path)

    print(f"Saving dimensions to: {dims_output_path}")
    all_dims_tensor = torch.tensor(all_dims, dtype=torch.int64)
    torch.save(all_dims_tensor, dims_output_path)


    print("Feature extraction complete!")

if __name__ == "__main__":
    extract_features() # You can pass config path and output dir as arguments if needed