"""Basic Predictive Coding Prototype

Implements a minimal hierarchical predictive coding network
operating on continuous vectors (as per project spec).

This is the starting implementation for Phase 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class PredictiveCodingLayer(nn.Module):
    """A single predictive coding layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(output_dim))

    def predict(self, higher_activity):
        """Generate prediction for lower layer from higher activity."""
        return torch.tanh(higher_activity @ self.W.T + self.b)


class HierarchicalPredictiveCoding(nn.Module):
    """Minimal 2-layer Hierarchical Predictive Coding model."""
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.layer1 = PredictiveCodingLayer(input_dim, hidden_dim)  # bottom -> middle
        self.layer2 = PredictiveCodingLayer(hidden_dim, hidden_dim) # middle -> top (abstract)
        
        # Simple recurrent state for temporal memory (middle layer)
        self.recurrent_state = torch.zeros(1, hidden_dim)

    def forward(self, z_t, num_inference_steps=5):
        """
        Inference: Settle activities by minimizing prediction error.
        z_t: continuous input vector (batch_size, input_dim)
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        
        # Initialize activities
        mu0 = z_t.clone()                    # bottom (input)
        mu1 = torch.zeros(batch_size, self.layer1.W.shape[0], device=device)  # middle
        mu2 = torch.zeros(batch_size, self.layer2.W.shape[0], device=device)  # top
        
        for _ in range(num_inference_steps):
            # Top-down predictions
            mu1_hat = self.layer2.predict(mu2)
            mu0_hat = self.layer1.predict(mu1)
            
            # Prediction errors
            e2 = mu2 - mu1_hat          # top level error
            e1 = mu1 - mu0_hat          # middle error
            e0 = mu0 - mu0_hat          # bottom error (vs input)
            
            # Simple error-driven updates (gradient descent on energy)
            mu2 = mu2 - 0.1 * e2
            mu1 = mu1 - 0.1 * (e1 + 0.05 * (mu1 - self.recurrent_state.to(device)) )  # temporal pull
            mu0 = mu0 - 0.1 * e0
        
        # Update recurrent state (simple exponential moving average for prototype)
        self.recurrent_state = 0.9 * self.recurrent_state + 0.1 * mu1.mean(dim=0, keepdim=True)
        
        return {
            'mu0': mu0,
            'mu1': mu1,
            'mu2': mu2,
            'e0': e0,
            'e1': e1,
            'e2': e2,
            'prediction': mu0_hat
        }

    def compute_energy(self, outputs):
        """Total prediction error energy."""
        return (outputs['e0']**2 + outputs['e1']**2 + outputs['e2']**2).mean()


def train_step(model, optimizer, z_sequence):
    """One training step on a sequence of continuous vectors."""
    total_loss = 0
    model.train()
    
    for z_t in z_sequence:
        z_t = z_t.unsqueeze(0)  # add batch dim
        outputs = model(z_t)
        loss = model.compute_energy(outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(z_sequence)


if __name__ == "__main__":
    print("=== Human Predictive AI - Basic Prototype ===")
    
    # Toy data: simple sequential patterns in continuous space
    input_dim = 64
    seq_len = 20
    batch_size = 4
    
    # Create dummy sequential data (random walks for testing)
    z_sequence = [torch.randn(input_dim) * 0.5 + torch.sin(torch.tensor(i) * 0.3) * 0.3 
                  for i in range(seq_len)]
    
    model = HierarchicalPredictiveCoding(input_dim=input_dim, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training on toy sequential data...")
    for epoch in tqdm(range(50)):
        loss = train_step(model, optimizer, z_sequence)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Avg Energy: {loss:.4f}")
    
    print("\nBasic prototype training complete!")
    print("Next: Expand hierarchy, improve temporal memory, add generation.")