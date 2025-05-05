
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pandas as pd
import random
from torchvision import transforms

from utils import preprocess_image, encode_text, CHARSET

class IAMHandwritingDataset(Dataset):
    def __init__(self, data_path, img_height=64, img_width=256, split='train', train_val_split=0.9, fraction=1.0):
        super().__init__()
        self.data_path = data_path
        self.img_height = img_height
        self.img_width = img_width
        self.split = split
        self.train_val_split = train_val_split
        self.fraction = fraction

        if self.split == 'train':
            self.augmentations = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
            ])
        else:
            self.augmentations = None

        self.image_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        annotation_file = os.path.join(self.data_path, 'words.txt')
        images_base_dir = os.path.join(self.data_path, 'words')

        valid_samples = []

        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue

                    image_id = parts[0]
                    transcription_status = parts[1]
                    transcription = parts[-1]

                    if transcription_status != 'ok':
                        continue

                    id_parts = image_id.split('-')
                    if len(id_parts) < 3:
                        continue

                    folder1 = id_parts[0]
                    folder2 = f"{id_parts[0]}-{id_parts[1]}"
                    image_filename = f"{image_id}.png"
                    image_path = os.path.join(images_base_dir, folder1, folder2, image_filename)

                    if os.path.exists(image_path) and all(c in CHARSET for c in transcription):
                        valid_samples.append({'path': image_path, 'label': transcription})

        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}. "
                                    f"Ensure 'words.txt' is in '{self.data_path}'.")
        except Exception as e:
            raise RuntimeError(f"Error reading annotation file {annotation_file}: {e}")

        if not valid_samples:
            raise ValueError(f"No valid samples found. Check data path ('{self.data_path}'), "
                             f"words.txt format, image file existence, and CHARSET.")

        # Apply fraction sampling ---
        random.seed(42) 
        random.shuffle(valid_samples)
        total_samples = int(len(valid_samples) * self.fraction)
        valid_samples = valid_samples[:total_samples]

        # Split data into train/val
        num_samples = len(valid_samples)
        num_train = int(num_samples * self.train_val_split)
        valid_samples.sort(key=lambda x: x['path'])

        if self.split == 'train':
            samples_to_use = valid_samples[:num_train]
        elif self.split == 'val' or self.split == 'test':
            samples_to_use = valid_samples[num_train:]
        else:
            raise ValueError(f"Invalid split name: {self.split}. Choose 'train', 'val', or 'test'.")

        self.image_paths = [sample['path'] for sample in samples_to_use]
        self.labels = [sample['label'] for sample in samples_to_use]

        print(f"Loaded {len(self.image_paths)} samples for split '{self.split}' (fraction={self.fraction}).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_text = self.labels[idx]

        image_tensor = preprocess_image(image_path, self.img_height, self.img_width)
        if image_tensor is None:
            print(f"Warning: Could not preprocess image {image_path}. Skipping.")
            return None

        if self.split == 'train' and self.augmentations:
            image_tensor = self.augmentations(image_tensor)

        label_encoded = encode_text(label_text)
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return image_tensor, label_tensor
    
def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences and padding.
    Filters out None items resulting from preprocessing errors.
    Args:
        batch (list): A list of tuples (image_tensor, label_tensor) from __getitem__.

    Returns:
        tuple: A tuple containing:
            - images (Tensor): Padded batch of image tensors (N, C, H, W).
            - targets (Tensor): Padded batch of encoded label tensors (N, max_label_len).
            - target_lengths (Tensor): Lengths of the original labels in the batch (N,).
            - input_lengths (Tensor): Not strictly needed for images here, but often needed for CTC with RNNs.
                                      We'll set it based on the model's output size later. Placeholder for now.
    """
    # Filter out None items (potential preprocessing errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return empty tensors or raise error if the entire batch failed
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])


    images, targets = zip(*batch)

    # Stack images (they should all have the same size C, H, W)
    images = torch.stack(images, 0) # (N, C, H, W)

    # Pad target sequences
    # Get lengths before padding
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    # Pad sequences to the max length in the batch
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0) # Use 0 for padding (consistent with blank)

    # Input lengths for CTC loss are typically derived from the model's output sequence length
    # For a CNN+RNN, this depends on the downsampling in the CNN part.
    # We will calculate this in the training loop based on model output.
    # For now, return a placeholder or calculate based on image width / CNN stride.
    # Let's calculate a dummy based on image width / assumed stride (e.g., 4)
    # This needs to match the actual output sequence length of the model!
    # We will override this in the training loop.
    input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long) # Placeholder


    return images, targets_padded, target_lengths, input_lengths


# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    DATA_DIR = './data' # CHANGE THIS if your data is elsewhere
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    BATCH_SIZE = 4 # Small batch size for testing

    # Check if data directory exists
    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, 'words.txt')):
         print(f"Error: Data directory '{DATA_DIR}' or 'words.txt' not found.")
         print("Please ensure the IAM dataset (words.txt, words/ folder) is placed correctly.")

    else:
        print("Attempting to load dataset...")
        try:
            # Create Datasets
            train_dataset = IAMHandwritingDataset(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, split='train')
            val_dataset = IAMHandwritingDataset(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, split='val')

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0, # Set to 0 for initial testing on Mac, can increase later
                pin_memory=True # Load data into memory for faster access
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True
            )

            # Test iterating through one batch
            print("\nTesting DataLoader iteration...")
            try:
                images, targets, target_lengths, input_lengths_placeholder = next(iter(train_loader))
                print(f"Batch - Images shape: {images.shape}")       # Should be [BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH]
                print(f"Batch - Targets shape: {targets.shape}")     # Should be [BATCH_SIZE, max_label_length]
                print(f"Batch - Target lengths: {target_lengths}")   # Should be [BATCH_SIZE]
                print(f"Batch - Input lengths (placeholder): {input_lengths_placeholder}") # Should be [BATCH_SIZE]

                # Decode first label in batch for verification
                first_label_indices = targets[0][:target_lengths[0]].tolist()
                # Import idx_to_char here if needed, or access via utils.idx_to_char
                from utils import idx_to_char
                decoded_label = "".join([idx_to_char.get(idx, '?') for idx in first_label_indices])
                print(f"Decoded first label in batch: '{decoded_label}'")


            except StopIteration:
                print("DataLoader is empty. This might happen if no valid samples were found.")
            except Exception as e:
                print(f"Error during DataLoader iteration test: {e}")


        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(e)
        except RuntimeError as e:
            print(f"Runtime error during dataset/loader creation: {e}")

