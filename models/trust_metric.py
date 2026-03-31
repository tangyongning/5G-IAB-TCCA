# models/trust_metric.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ReliabilityTrustMetric(nn.Module):
    """
    Reliability-Oriented Trust Metric Module (Section III-C)
    
    Implements Eq. (7)-(9):
    - Trust score: T_v(t) = γ₁·C_v(t) + γ₂·S_v(t) + γ₃·H_v(t)
    - Structural consistency: C_v(t)
    - Observational support: S_v(t) = ||y_v(t) - y_normal||
    - Temporal stability: H_v(t)
    """
    
    def __init__(self,
                 num_nodes: int,
                 adjacency_matrix: torch.Tensor,
                 gamma1: float = 0.4,
                 gamma2: float = 0.3,
                 gamma3: float = 0.3):
        super(ReliabilityTrustMetric, self).__init__()
        
        self.num_nodes = num_nodes
        
        # Learnable weighting coefficients
        self.gamma1 = nn.Parameter(torch.tensor(gamma1))
        self.gamma2 = nn.Parameter(torch.tensor(gamma2))
        self.gamma3 = nn.Parameter(torch.tensor(gamma3))
        
        # Register adjacency matrix
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        
        # Baseline QoS parameters (learnable)
        self.baseline_qos = nn.Parameter(torch.zeros(4))  # latency, loss, throughput, availability
        self.baseline_std = nn.Parameter(torch.ones(4))
        
        # Uncertainty estimation (Monte Carlo dropout)
        self.dropout = nn.Dropout(p=0.3)
        
    def compute_structural_consistency(self,
                                        fault_probs: torch.Tensor,
                                        threshold: float = 0.5) -> torch.Tensor:
        """
        Compute structural consistency C_v(t) (Eq. 8)
        
        Measures whether inferred relationships are consistent with
        topology-constrained causality
        
        Args:
            fault_probs: (batch_size, num_nodes, 1)
            threshold: fault probability threshold
            
        Returns:
            consistency: (batch_size, num_nodes, 1)
        """
        batch_size = fault_probs.size(0)
        
        # Get parent nodes for each node
        parents = self.adjacency_matrix.T  # (N, N)
        
        # Check consistency: if parent is faulty, child should show degradation
        parent_faulty = (fault_probs > threshold).float()  # (B, N, 1)
        
        # Compute consistency for each node
        consistency_scores = []
        
        for i in range(self.num_nodes):
            parent_indices = torch.where(parents[i] > 0)[0]
            
            if len(parent_indices) == 0:
                # Root nodes have perfect consistency
                consistency_scores.append(torch.ones(batch_size, 1))
            else:
                # Check if parent faults imply child faults
                parent_faults = parent_faulty[:, parent_indices, :]  # (B, num_parents, 1)
                child_fault = fault_probs[:, i:i+1, :]  # (B, 1, 1)
                
                # Consistency: parent faulty → child faulty
                consistent = (parent_faults.mean(dim=1) <= child_fault + 0.3).float()
                consistency_scores.append(consistent)
        
        consistency = torch.cat(consistency_scores, dim=1).unsqueeze(-1)  # (B, N, 1)
        
        return consistency
    
    def compute_observational_support(self,
                                       qos_observations: torch.Tensor) -> torch.Tensor:
        """
        Compute observational support S_v(t) (Eq. 9)
        
        S_v(t) = ||y_v(t) - y_normal||
        
        Args:
            qos_observations: (batch_size, num_nodes, 4) - [latency, loss, throughput, availability]
            
        Returns:
            support: (batch_size, num_nodes, 1)
        """
        # Normalize observations
        normalized_obs = (qos_observations - self.baseline_qos) / (self.baseline_std + 1e-8)
        
        # Compute Mahalanobis-like distance
        # S_v(t) = ||y_v(t) - y_normal||
        support = torch.norm(normalized_obs, dim=-1, keepdim=True)  # (B, N, 1)
        
        # Normalize to [0, 1]
        support = torch.sigmoid(support)
        
        return support
    
    def compute_temporal_stability(self,
                                    fault_history: torch.Tensor,
                                    window_size: int = 5) -> torch.Tensor:
        """
        Compute temporal stability H_v(t)
        
        Measures historical consistency of fault predictions
        
        Args:
            fault_history: (batch_size, num_nodes, window_size)
            window_size: number of time steps to consider
            
        Returns:
            stability: (batch_size, num_nodes, 1)
        """
        # Compute variance over time window
        mean_fault = fault_history.mean(dim=-1, keepdim=True)  # (B, N, 1)
        variance = ((fault_history - mean_fault) ** 2).mean(dim=-1, keepdim=True)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + variance)  # (B, N, 1)
        
        return stability
    
    def compute_uncertainty(self,
                            fault_probs: torch.Tensor,
                            num_samples: int = 50) -> torch.Tensor:
        """
        Compute prediction uncertainty using Monte Carlo Dropout
        
        U_v(t) = sqrt(1/M · Σ(p_v^(m)(t) - p̄_v(t))²)
        
        Args:
            fault_probs: (batch_size, num_nodes, 1)
            num_samples: number of MC dropout samples
            
        Returns:
            uncertainty: (batch_size, num_nodes, 1)
        """
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(num_samples):
            # Apply dropout to get varied predictions
            dropped_probs = self.dropout(fault_probs)
            samples.append(dropped_probs)
        
        samples = torch.cat(samples, dim=-1)  # (B, N, num_samples)
        
        # Compute variance
        mean_prob = samples.mean(dim=-1, keepdim=True)  # (B, N, 1)
        variance = ((samples - mean_prob) ** 2).mean(dim=-1, keepdim=True)
        
        uncertainty = torch.sqrt(variance + 1e-8)  # (B, N, 1)
        
        self.eval()
        
        return uncertainty
    
    def forward(self,
                fault_probs: torch.Tensor,
                qos_observations: torch.Tensor,
                fault_history: Optional[torch.Tensor] = None,
                return_uncertainty: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Compute trust score T_v(t) (Eq. 7)
        
        Args:
            fault_probs: (batch_size, num_nodes, 1)
            qos_observations: (batch_size, num_nodes, 4)
            fault_history: (batch_size, num_nodes, window_size)
            return_uncertainty: whether to return uncertainty estimate
            
        Returns:
            trust_score: (batch_size, num_nodes, 1)
            components: dict with C_v, S_v, H_v
            uncertainty: (batch_size, num_nodes, 1)
        """
        # Compute structural consistency C_v(t)
        structural_consistency = self.compute_structural_consistency(fault_probs)
        
        # Compute observational support S_v(t)
        observational_support = self.compute_observational_support(qos_observations)
        
        # Compute temporal stability H_v(t)
        if fault_history is not None:
            temporal_stability = self.compute_temporal_stability(fault_history)
        else:
            temporal_stability = torch.ones_like(fault_probs) * 0.5
        
        # Compute trust score T_v(t) = γ₁·C_v + γ₂·S_v + γ₃·H_v (Eq. 7)
        # Normalize gammas
        gamma_sum = self.gamma1 + self.gamma2 + self.gamma3 + 1e-8
        trust_score = (
            (self.gamma1 / gamma_sum) * structural_consistency +
            (self.gamma2 / gamma_sum) * observational_support +
            (self.gamma3 / gamma_sum) * temporal_stability
        )
        
        components = {
            'structural_consistency': structural_consistency,
            'observational_support': observational_support,
            'temporal_stability': temporal_stability
        }
        
        if return_uncertainty:
            uncertainty = self.compute_uncertainty(fault_probs)
            # Penalize high uncertainty in trust score
            trust_score = trust_score - 0.1 * uncertainty
            return trust_score, components, uncertainty
        
        return trust_score, components, None
