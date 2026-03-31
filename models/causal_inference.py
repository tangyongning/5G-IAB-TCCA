# models/causal_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class TopologyConstrainedCausalInference(nn.Module):
    """
    Topology-Constrained Causal Inference Module (Section III-B)
    
    Implements Eq. (4)-(6):
    - Aggregated upstream influence: φ_v(t) = Σ_{u∈P(v)} w_{u,v} · f_u(t)
    - Fault likelihood: s_v(t) = σ(α · f_v(t) + β · φ_v(t))
    - Topology constraint: w_{v,u} = 0 if (u,v) ∉ E
    """
    
    def __init__(self, 
                 num_nodes: int,
                 adjacency_matrix: torch.Tensor,
                 hidden_dim: int = 64,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        super(TopologyConstrainedCausalInference, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
        # Register adjacency matrix as buffer (not trainable)
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        
        # Learnable influence weights (initialized uniformly)
        self.influence_weights = nn.Parameter(
            torch.ones(num_nodes, num_nodes) / num_nodes
        )
        
        # Local fault feature encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4 QoS metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for dynamic weight learning
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        
    def mask_weights(self):
        """Enforce topology constraint: w_{v,u} = 0 if (u,v) ∉ E (Eq. 6)"""
        masked_weights = self.influence_weights * self.adjacency_matrix.T
        # Normalize weights to sum to 1 for each node
        row_sum = masked_weights.sum(dim=1, keepdim=True) + 1e-8
        masked_weights = masked_weights / row_sum
        return masked_weights
    
    def compute_upstream_influence(self, 
                                    fault_states: torch.Tensor,
                                    node_features: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated upstream influence φ_v(t) (Eq. 4)
        
        Args:
            fault_states: (batch_size, num_nodes, 1) - fault probabilities
            node_features: (batch_size, num_nodes, feature_dim) - QoS features
            
        Returns:
            upstream_influence: (batch_size, num_nodes, 1)
        """
        batch_size = fault_states.size(0)
        
        # Get masked and normalized weights
        weights = self.mask_weights()  # (num_nodes, num_nodes)
        
        # Compute weighted sum of upstream fault states
        # φ_v(t) = Σ_{u∈P(v)} w_{u,v} · f_u(t)
        upstream_influence = torch.matmul(fault_states.transpose(1, 2), 
                                          weights).transpose(1, 2)
        
        return upstream_influence  # (batch_size, num_nodes, 1)
    
    def compute_attention_weights(self, 
                                   node_features: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute dynamic attention weights for influence modeling
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim)
            mask: (num_nodes, num_nodes) - topology mask
            
        Returns:
            attention_weights: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Compute attention scores
        query = self.attention_query(node_features)  # (B, N, H)
        key = self.attention_key(node_features)  # (B, N, H)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        
        # Apply topology mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)  # (B, N, N)
        
        return attention_weights
    
    def forward(self, 
                fault_states: torch.Tensor,
                node_features: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for causal inference
        
        Args:
            fault_states: (batch_size, num_nodes, 1)
            node_features: (batch_size, num_nodes, 4) - QoS metrics
            
        Returns:
            fault_likelihood: (batch_size, num_nodes, 1) - s_v(t)
            upstream_influence: (batch_size, num_nodes, 1) - φ_v(t)
        """
        batch_size = fault_states.size(0)
        
        # Encode local features
        local_features = self.local_encoder(node_features)  # (B, N, H)
        
        # Compute local fault contribution
        local_fault = fault_states  # (B, N, 1)
        
        # Compute upstream influence (Eq. 4)
        upstream_influence = self.compute_upstream_influence(
            fault_states, node_features
        )  # (B, N, 1)
        
        # Compute fault likelihood (Eq. 5)
        # s_v(t) = σ(α · f_v(t) + β · φ_v(t))
        fault_likelihood = torch.sigmoid(
            self.alpha * local_fault + self.beta * upstream_influence
        )
        
        if return_weights:
            weights = self.mask_weights()
            return fault_likelihood, upstream_influence, weights
        
        return fault_likelihood, upstream_influence
