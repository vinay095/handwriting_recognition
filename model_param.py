import torch

# Load the state dictionary
state_dict = torch.load('./saved_models/best_model_acc.pth', map_location='cpu')

# Print all parameter names and their shapes
for param_name, param_tensor in state_dict.items():
    print(f"{param_name}: {param_tensor.shape}")