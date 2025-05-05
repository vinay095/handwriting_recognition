# utils.py
import torch
import numpy as np
from PIL import Image
import string

# --- Character Mapping ---

# Define the character set. Add any other characters potentially present in your dataset.
# Ensure the 'blank' token is handled (usually implicitly by CTCLoss at index 0)
CHARSET = string.ascii_letters + string.digits + " .,;:!?'\"-()&" # Example charset
# CHARSET = """!"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """ # Alternative wider charset if needed

# Create character-to-index and index-to-character mappings
# Reserve index 0 for the CTC blank token
char_to_idx = {char: i + 1 for i, char in enumerate(CHARSET)}
idx_to_char = {i + 1: char for i, char in enumerate(CHARSET)}
num_classes = len(CHARSET) + 1  # Add 1 for the blank token

# --- Encoding and Decoding ---

def encode_text(text):
    """Encodes a text string into a list of integers based on char_to_idx."""
    try:
        encoded = [char_to_idx[char] for char in text if char in char_to_idx] # Ignore chars not in CHARSET
        return encoded
    except KeyError as e:
        print(f"Warning: Character '{e}' not in CHARSET. Ignoring.")
        # Fallback: encode only known characters
        encoded = [char_to_idx[char] for char in text if char in char_to_idx]
        return encoded


def ctc_greedy_decode(log_probs, input_lengths):
    """
    Performs greedy CTC decoding on log probabilities.
    Args:
        log_probs (Tensor): Log probabilities output by the model (T, N, C) or (N, T, C) if batch_first=True.
                           Assumes batch_first=False (Time dimension first).
        input_lengths (Tensor): Lengths of the sequences in the batch (N,).

    Returns:
        list[str]: Decoded text strings for each item in the batch.
    """
    decoded_texts = []
    # Assuming log_probs shape is (Time, Batch, Classes)
    # Get the most likely character index at each time step
    # Argmax returns indices, shape (Time, Batch)
    pred_indices = torch.argmax(log_probs.detach(), dim=2)
    # Transpose to (Batch, Time) for easier iteration
    pred_indices = pred_indices.transpose(0, 1).cpu().numpy()
    input_lengths = input_lengths.cpu().numpy()

    for i in range(pred_indices.shape[0]): # Iterate through batch
        seq_len = input_lengths[i]
        raw_sequence = pred_indices[i, :seq_len]

        # Greedy decoding: Remove consecutive duplicates and then blanks (index 0)
        decoded_sequence = []
        last_char_idx = 0 # Use 0 for blank
        for char_idx in raw_sequence:
            if char_idx != last_char_idx: # If different from last char
                if char_idx != 0: # And not blank
                    decoded_sequence.append(char_idx)
                last_char_idx = char_idx

        # Convert indices back to characters
        try:
            decoded_text = "".join([idx_to_char[idx] for idx in decoded_sequence])
            decoded_texts.append(decoded_text)
        except KeyError as e:
             print(f"Warning: Decoded index '{e}' not found in idx_to_char map during decoding. Skipping character.")
             # Attempt partial decoding
             partial_text = ""
             for idx in decoded_sequence:
                 if idx in idx_to_char:
                     partial_text += idx_to_char[idx]
             decoded_texts.append(partial_text + " [Decoding Error]")


    return decoded_texts

def ctc_beam_search_decode(log_probs, input_lengths, beam_width=3):
    """
    Performs a simple beam search decoding on log probabilities.

    Args:
        log_probs (Tensor): Log probabilities output by the model (T, N, C).
        input_lengths (Tensor): Lengths of the sequences in the batch (N,).
        beam_width (int): Number of beams to keep during decoding.

    Returns:
        list[str]: Beam-decoded text strings.
    """
    log_probs = log_probs.cpu().detach()  # (T, N, C)
    T, N, C = log_probs.size()
    input_lengths = input_lengths.cpu()

    results = []

    for b in range(N):  # For each item in the batch
        beams = [([], 0.0)]  # (sequence, score)
        seq_len = input_lengths[b]

        for t in range(seq_len):
            next_beams = []
            for seq, score in beams:
                for c in range(C):
                    new_seq = seq + [c]
                    new_score = score + log_probs[t, b, c].item()
                    next_beams.append((new_seq, new_score))

            # Keep top beam_width sequences
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:beam_width]

        # Deduplicate & remove blanks (index 0) like in CTC
        best_seq = beams[0][0]
        decoded = []
        last = None
        for i in best_seq:
            if i != 0 and i != last:
                decoded.append(i)
            last = i

        decoded_text = "".join([idx_to_char.get(idx, '') for idx in decoded])
        results.append(decoded_text)

    return results
# --- Image Preprocessing ---

def preprocess_image(image_path, img_height=64, img_width=256):
    """
    Loads an image, converts to grayscale, resizes, normalizes, and converts to tensor.
    Args:
        image_path (str): Path to the image file.
        img_height (int): Target height for resizing.
        img_width (int): Target width for resizing.

    Returns:
        Tensor: Preprocessed image tensor (C, H, W).
    """
    try:
        img = Image.open(image_path).convert('L')  # Load and convert to grayscale

        # Resize maintaining aspect ratio (optional, can also just resize directly)
        # Calculate aspect ratio
        w, h = img.size
        target_w = int(w * img_height / h)
        target_w = min(target_w, img_width) # Cap width if aspect ratio makes it too wide

        # Resize using LANCZOS filter for better quality
        img = img.resize((target_w, img_height), Image.Resampling.LANCZOS)

        # Create a new image with target width and paste the resized image
        new_img = Image.new('L', (img_width, img_height), color=255) # White background
        new_img.paste(img, (0, 0))
        img = new_img

        # Convert to numpy array and normalize to [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0

        # Add channel dimension (C, H, W)
        img = np.expand_dims(img, axis=0)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(img)

        # Optional: Normalize further (e.g., mean/std normalization if needed)
        # Example: tensor = (tensor - 0.5) / 0.5

        return tensor
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage/testing
    print(f"Charset: '{CHARSET}'")
    print(f"Number of classes (including blank): {num_classes}")

    test_text = "Hello 123"
    encoded = encode_text(test_text)
    print(f"Encoded '{test_text}': {encoded}")

    # Example decoding (requires dummy log_probs and lengths)
    # Shape: (Time, Batch, Classes)
    dummy_log_probs = torch.randn(50, 1, num_classes) # T=50, N=1, C=num_classes
    dummy_log_probs[:, 0, char_to_idx['H']] = 10 # Make 'H' likely at step 0
    dummy_log_probs[:, 0, char_to_idx['e']] = 10 # Make 'e' likely at step 1
    dummy_log_probs[:, 0, char_to_idx['l']] = 10 # Make 'l' likely at step 2
    dummy_log_probs[:, 0, char_to_idx['l']] = 10 # Make 'l' likely at step 3 (duplicate)
    dummy_log_probs[:, 0, 0] = 10             # Make blank likely at step 4
    dummy_log_probs[:, 0, char_to_idx['o']] = 10 # Make 'o' likely at step 5
    dummy_lengths = torch.tensor([50]) # Length of sequence for batch item 0 is 50

    decoded = ctc_greedy_decode(dummy_log_probs, dummy_lengths)
    print(f"Example Greedy Decode: {decoded}")

    # Test preprocessing
    # Create a dummy white image file
    try:
        dummy_img = Image.new('L', (150, 50), color=255)
        dummy_img.save("dummy_test_image.png")
        preprocessed = preprocess_image("dummy_test_image.png")
        if preprocessed is not None:
            print(f"Preprocessed image tensor shape: {preprocessed.shape}") # Should be [1, 64, 256]
        import os
        os.remove("dummy_test_image.png")
    except Exception as e:
        print(f"Could not test preprocessing: {e}")

