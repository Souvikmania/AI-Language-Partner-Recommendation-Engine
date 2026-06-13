import torch
import torch.nn as nn

class CompatibilityModel(nn.Module):
    def __init__(self, input_dim=12):
        super(CompatibilityModel, self).__init__()
        
        # Enhanced Architecture for 12 features:
        # Input (12 features):
        #   - 8 base features: [A_known, A_learning, A_skill, A_streak, B_known, B_learning, B_skill, B_streak]
        #   - 4 derived features: [streak_diff, skill_diff, is_complementary, is_peer]
        #
        # Dense layers (64 → 32 → 16) with ReLU activations
        # Output (1 unit) with Sigmoid for score between 0 and 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output score between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    model = CompatibilityModel(input_dim=12)
    print(model)
    # Test with dummy data (12 features)
    dummy_input = torch.randn(1, 12)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Sample output (compatibility score): {output.item():.4f}")
