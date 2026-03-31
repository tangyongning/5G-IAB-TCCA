# evaluate.py
import torch
import numpy as np
from models.tcca_framework import TCCAFramework

def evaluate_model(model_path: str,
                   test_data: tuple,
                   device: str = 'cuda') -> dict:
    """
    Evaluate trained TCCA model
    
    Args:
        model_path: path to saved model checkpoint
        test_data: (adj_matrix, X_test, y_test)
        device: evaluation device
        
    Returns:
        metrics: dict with evaluation metrics
    """
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract data
    adj_matrix, X_test, y_test = test_data
    adj_matrix = adj_matrix.to(device)
    
    # Initialize model
    model = TCCAFramework(
        num_nodes=adj_matrix.size(0),
        adjacency_matrix=adj_matrix,
        input_dim=4,
        hidden_dim=64
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to tensors
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Evaluate
    with torch.no_grad():
        output = model(X_test)
        
        fault_probs = output['fault_probs'][:, -1, :, 0]
        trust_scores = output['trust_scores'][:, -1, :, 0]
        predictions = (fault_probs > 0.5).float()
        
        # Compute metrics
        tp = ((predictions == 1) & (y_test[:, -1, :, 0] == 1)).sum().item()
        fp = ((predictions == 1) & (y_test[:, -1, :, 0] == 0)).sum().item()
        tn = ((predictions == 0) & (y_test[:, -1, :, 0] == 0)).sum().item()
        fn = ((predictions == 0) & (y_test[:, -1, :, 0] == 1)).sum().item()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        far = fp / (fp + tn + 1e-8) * 100
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_alarm_rate': far,
            'checkpoint_epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss']
        }
    
    return metrics

if __name__ == '__main__':
    # Example usage
    from utils.data_loader import generate_synthetic_data
    
    adj_matrix, _, _, X_test, y_test = generate_synthetic_data(
        num_nodes=100,
        train_samples=0,
        val_samples=1000
    )
    
    metrics = evaluate_model(
        model_path='best_tcca_model.pth',
        test_data=(adj_matrix, X_test, y_test)
    )
    
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
