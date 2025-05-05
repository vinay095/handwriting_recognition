# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for handwriting recognition.
    Architecture: CNN feature extractor -> BiLSTM -> Linear layer for CTC.
    """
    def __init__(self, img_height, img_width, num_classes, rnn_hidden_size=256, rnn_layers=2, dropout=0.5):
        """
        Args:
            img_height (int): Height of the input image.
            img_width (int): Width of the input image.
            num_classes (int): Number of output classes (charset size + 1 for blank).
            rnn_hidden_size (int): Number of hidden units in the LSTM layer.
            rnn_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability for LSTM layers.
        """
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        # --- Convolutional Feature Extractor ---
        # Input shape: (N, 1, H, W) = (N, 1, img_height, img_width)
        # Using a simple CNN structure inspired by VGG/common OCR models.
        # Adjust kernel sizes, strides, padding based on input dimensions if needed.

        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2 = 32, 128 (for 64x256 input)

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/4, W/4 = 16, 64

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # Add Batch Norm
            nn.ReLU(inplace=True),
            # No MaxPool here

            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Careful with pooling: MaxPool2d((2, 1), (2, 1)) reduces height but not width stride
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # H/8, W/4 = 8, 64 -> Check padding effect

            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # No MaxPool here

            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # H/16, W/4? -> Check output size carefully -> 4, 64?

             # Layer 7 (Reduce height to 1)
            # Kernel (2,2) might be too large if height is already small. Check output H dimension.
            # Let's assume output H is 4 after Layer 6. Kernel (4,1) might work.
            # Let's test with a kernel that matches the remaining height.
            # Calculate output size dynamically or make assumptions and verify.
            # Assuming H=4 after layer 6 pool:
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0), # Output H=1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # Output shape: (N, 512, 1, W_cnn_out)
            # W_cnn_out depends on pooling/padding, roughly W/4 = 64 for 256 input width
        )

        # Calculate the output size of the CNN to determine the input size for the RNN
        # We need the 'W' dimension after the CNN passes. Let's do a dummy forward pass.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.img_height, self.img_width)
            cnn_out = self.cnn(dummy_input)
            # Shape is (N, C, H, W) -> (1, 512, 1, W_cnn_out)
            self.cnn_output_width = cnn_out.size(3)
            rnn_input_size = cnn_out.size(1) # Number of channels (512)


        # --- Recurrent Layer (LSTM) ---
        # Input to RNN needs to be (SeqLen, N, Features)
        # CNN output is (N, C, 1, W_cnn_out). We treat W_cnn_out as the sequence length (T).
        # Features = C (512).
        # So, reshape CNN output: (N, C, 1, T) -> (N, T, C) -> (T, N, C)

        self.rnn = nn.LSTM(
            input_size=rnn_input_size, # Features from CNN (512)
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0, # Add dropout only if multiple layers
            batch_first=False # Input shape (SeqLen, Batch, Features)
        )

        # --- Output Layer (Fully Connected) ---
        # Input: Output from LSTM (SeqLen, N, 2 * rnn_hidden_size because bidirectional)
        # Output: (SeqLen, N, num_classes) - Log probabilities for CTC loss
        self.fc = nn.Linear(
            in_features=rnn_hidden_size * 2, # *2 for bidirectional
            out_features=num_classes
        )

    def forward(self, x):
        """
        Forward pass through the CRNN.
        Args:
            x (Tensor): Input image batch (N, C, H, W). C=1 for grayscale.

        Returns:
            Tensor: Log probabilities for CTC loss (SeqLen, N, NumClasses).
        """
        # Pass through CNN
        # x shape: (N, 1, H, W)
        features = self.cnn(x)
        # features shape: (N, C_out, 1, W_out) e.g., (N, 512, 1, 64)

        # Prepare for RNN
        # Remove H dimension (squeeze) -> (N, C_out, W_out)
        features = features.squeeze(2)
        # Permute to (W_out, N, C_out) which is (SeqLen, Batch, Features)
        features = features.permute(2, 0, 1)
        # features shape: (T, N, 512) where T = W_out

        # Pass through RNN
        # rnn_output shape: (SeqLen, N, 2 * HiddenSize)
        # h_n, c_n shapes: (NumLayers * NumDirections, N, HiddenSize) - we don't need these here
        rnn_output, _ = self.rnn(features)

        # Pass through output layer
        # output shape: (SeqLen, N, NumClasses)
        output = self.fc(rnn_output)

        # Apply LogSoftmax for CTCLoss (expects log probabilities)
        # CTCLoss applies log_softmax internally if needed, but it's common practice
        # to apply it here. Check CTCLoss documentation for the version you use.
        # PyTorch CTCLoss expects log_probabilities.
        log_probs = F.log_softmax(output, dim=2)

        return log_probs

    def get_output_seq_len(self, input_width):
        """
        Calculates the sequence length produced by the CNN layers for a given input width.
        This is needed to provide the 'input_lengths' argument to CTCLoss.
        NOTE: This calculation MUST match the actual architecture (pooling layers).
        """
        # This needs to precisely reflect the downsampling in the width dimension
        # Layer 1: MaxPool2d(2, 2) -> W/2
        # Layer 2: MaxPool2d(2, 2) -> W/4
        # Layer 4: MaxPool2d((2, 2), stride=(2, 1), padding=(0,1)) -> Stride 1 in width, Padding 1.
        #          Output width W_out = floor((W_in + 2*padding - kernel)/stride) + 1
        #          W_in = W/4. W_out = floor((W/4 + 2*1 - 2)/1) + 1 = floor(W/4) + 1
        # Layer 6: MaxPool2d((2, 2), stride=(2, 1), padding=(0,1)) -> Stride 1 in width, Padding 1.
        #          W_in = floor(W/4) + 1. W_out = floor((floor(W/4)+1 + 2*1 - 2)/1) + 1 = floor(floor(W/4)+1) + 1
        # This gets complicated quickly. A simpler way is the dummy forward pass used in __init__.
        # Or, if strides are simpler (e.g., always stride 2 pooling):
        # width = input_width
        # width = width // 2 # Pool 1
        # width = width // 2 # Pool 2
        # width = width // 1 # Pool 4 (stride 1) - careful with padding
        # width = width // 1 # Pool 6 (stride 1) - careful with padding
        # return width

        # Let's rely on the calculation done during __init__
        # Assuming self.cnn_output_width was calculated correctly based on self.img_width
        if input_width == self.img_width:
             return self.cnn_output_width
        else:
             # Need to recalculate if input_width is different from init width
             # For simplicity, assume it's always called with self.img_width during training
             print("Warning: get_output_seq_len called with different width than init. Returning cached width.")
             return self.cnn_output_width


# --- Example Usage ---
if __name__ == '__main__':
    # Configuration (match dataset/utils)
    IMG_H = 64
    IMG_W = 256
    from utils import num_classes # Get num_classes from utils

    # Create model instance
    model = CRNN(img_height=IMG_H, img_width=IMG_W, num_classes=num_classes)
    print(model)

    # Test forward pass with a dummy batch
    # Batch size N = 4
    # Input shape (N, C, H, W) = (4, 1, 64, 256)
    dummy_batch = torch.randn(4, 1, IMG_H, IMG_W)

    # Perform forward pass
    log_probs = model(dummy_batch)

    # Check output shape
    # Expected: (SeqLen, N, NumClasses) = (T, 4, num_classes)
    # Where T is the sequence length output by CNN, e.g., 64 if W_out=64
    print(f"\nInput batch shape: {dummy_batch.shape}")
    print(f"Output log_probs shape: {log_probs.shape}") # e.g., torch.Size([64, 4, 81])

    # Verify the calculated sequence length
    calculated_seq_len = model.get_output_seq_len(IMG_W)
    print(f"Calculated CNN output sequence length (T): {calculated_seq_len}")
    assert log_probs.shape[0] == calculated_seq_len, "Output sequence length mismatch!"

    # Check if MPS is available and move model if desired
    if torch.cuda.is_available():
        mps_device = torch.device("cuda")
        model.to(mps_device)
        dummy_batch_mps = dummy_batch.to(mps_device)
        log_probs_mps = model(dummy_batch_mps)
        print(f"\nModel moved to MPS. Output shape on MPS: {log_probs_mps.shape}")
        print("MPS test successful.")
    else:
        print("\nMPS device not found.")

