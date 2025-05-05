# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
from jiwer import wer, cer

# Import local modules
from dataset import IAMHandwritingDataset, collate_fn
from model import CRNN
from utils import num_classes, ctc_greedy_decode,ctc_beam_search_decode # Import num_classes and decoder

def validate(model, dataloader, criterion, device):
    """Calculates validation loss and accuracy (using greedy decoding)."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct_chars = 0
    total_chars = 0
    total_correct_words = 0
    total_words = 0

    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images, targets, target_lengths, _ = batch # Ignore placeholder input_lengths

            # Move data to the appropriate device (CPU or MPS)
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Forward pass
            log_probs = model(images) # Output shape: (SeqLen, N, NumClasses)

            # Calculate the actual input lengths for CTC loss based on model output
            # Shape: (N,) - filled with the sequence length from the model (T)
            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long).to(device)

            # Calculate CTC loss
            # Input: (T, N, C), Targets: (N, S), InputLengths: (N,), TargetLengths: (N,)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item() * images.size(0) # Accumulate loss weighted by batch size

            # --- Calculate Accuracy (Optional but useful) ---
            # Use greedy decoding for validation accuracy check
            from utils import ctc_greedy_decode, ctc_beam_search_decode
            use_beam = True  # Set this to False to revert to greedy decoding

            if use_beam:
                decoded_preds = ctc_beam_search_decode(log_probs, input_lengths, beam_width=3)
            else:
                decoded_preds = ctc_greedy_decode(log_probs, input_lengths)         

            # Decode ground truth targets
            # Need idx_to_char mapping from utils
            from utils import idx_to_char
            decoded_targets = []
            for i in range(targets.size(0)):
                target_indices = targets[i][:target_lengths[i]].cpu().numpy()
                decoded_text = "".join([idx_to_char.get(idx, '') for idx in target_indices if idx != 0]) # Skip padding/blank
                decoded_targets.append(decoded_text)

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_targets)

            # Compute CER/WER
            char_error = cer(all_labels, all_preds)
            word_error = wer(all_labels, all_preds)

            # Compare decoded strings (simple exact match word accuracy)
            for pred, target in zip(decoded_preds, decoded_targets):
                if pred == target:
                    total_correct_words += 1
                total_words += 1
                # Can also calculate Character Error Rate (CER) if needed (more involved)


    avg_loss = total_loss / len(dataloader.dataset)
    word_accuracy = (total_correct_words / total_words) * 100 if total_words > 0 else 0

    # Print first few validation predictions vs labels
    print("\n--- Validation Examples ---")
    for i in range(min(5, len(all_preds))):
         print(f"Pred : '{all_preds[i]}'")
         print(f"Label: '{all_labels[i]}'")
         print("-" * 10)

    model.train() # Set model back to training mode
    return avg_loss, word_accuracy, char_error, word_error


def train(args):
    """Main training loop."""

    # --- Device Setup ---
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # --- Create Datasets and DataLoaders ---
    print("Loading datasets...")
    try:
        train_dataset = IAMHandwritingDataset(
            data_path=args.data_path,
            img_height=args.img_height,
            img_width=args.img_width,
            split='train',
            fraction=args.fraction
        )
        val_dataset = IAMHandwritingDataset(
            data_path=args.data_path,
            img_height=args.img_height,
            img_width=args.img_width,
            split='val',
            fraction=0.1
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error initializing dataset: {e}")
        return # Exit if dataset loading fails

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device != torch.device('cpu') else False # Pin memory if using GPU/MPS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device != torch.device('cpu') else False
    )
    print("Datasets loaded.")

    # --- Model Initialization ---
    print("Initializing model...")
    model = CRNN(
        img_height=args.img_height,
        img_width=args.img_width,
        num_classes=num_classes, # Get from utils
        rnn_hidden_size=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Loss Function and Optimizer ---
    # CTCLoss: reduction='mean' averages loss per batch item, zero_infinity=True helps stability
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
    #  Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True) # Create directory for saving models

    for epoch in range(1, args.epochs + 1):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for batch in progress_bar:
            # Unpack batch and move to device
            images, targets, target_lengths, _ = batch # Ignore placeholder input_lengths
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            log_probs = model(images) # Output shape: (SeqLen, N, NumClasses)

            # Calculate actual input lengths for CTC loss
            # Shape: (N,) - filled with the sequence length from the model (T)
            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long).to(device)

            # Calculate CTC loss
            # Input: (T, N, C), Targets: (N, S), InputLengths: (N,), TargetLengths: (N,)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Handle potential NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered in epoch {epoch}. Skipping batch.")
                # Optional: Add gradient clipping here if explosions are suspected
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                continue # Skip backward pass and optimizer step for this batch

            # Backward pass
            loss.backward()

            #  Gradient clipping (helps prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item() * images.size(0) # Accumulate loss weighted by batch size

            # Update progress bar description
            progress_bar.set_postfix(loss=loss.item())


        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)

        # --- Validation ---
        val_loss, val_acc, val_cer, val_wer = validate(model, val_loader, criterion, device)

        #scheduler.step() # Step the scheduler after validation
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}")
        print(f"  Val Acc   : {val_acc:.2f}% (Exact Word Match)")
        print(f"  CER       : {val_cer:.4f}")
        print(f"  WER       : {val_wer:.4f}")

        # Optional: Adjust learning rate using scheduler
        # scheduler.step()

        # --- Save Model Checkpoints ---
        # Save the last model
        last_model_path = os.path.join(args.save_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)

        # Save the best model based on validation loss or accuracy
        # Using accuracy here
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model_acc.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (by accuracy) to {best_model_path}")

        # Or save based on loss
        if val_loss < best_val_loss:
             best_val_loss = val_loss
             best_model_path_loss = os.path.join(args.save_dir, 'best_model_loss.pth')
             torch.save(model.state_dict(), best_model_path_loss)
             print(f"  -> Saved best model (by loss) to {best_model_path_loss}")


    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved in '{args.save_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CRNN for Handwriting Recognition")

    # Data args
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the IAM dataset directory')
    parser.add_argument('--img_height', type=int, default=64, help='Image height')
    parser.add_argument('--img_width', type=int, default=320, help='Image width')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of dataset to load (e.g., 0.1 = 10%)')
    # Model args
    parser.add_argument('--rnn_hidden', type=int, default=256, help='RNN hidden size')
    parser.add_argument('--rnn_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')

    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping max norm') # Optional
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--force_cpu', action='store_true', help='Force use CPU even if MPS is available')


    args = parser.parse_args()

    # Print config
    print("--- Training Configuration ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 30)

    train(args)

    #export PYTORCH_ENABLE_MPS_FALLBACK=1


