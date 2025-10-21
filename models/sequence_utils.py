#!/usr/bin/env python3
"""
Sequence utilities for protein VAE model.
Handles amino acid encoding/decoding and sequence prediction.
"""

import torch
import torch.nn.functional as F

# Standard 20 amino acids in alphabetical order
AA_ORDER = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Create mapping from amino acid to index
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ORDER)}

def sequence_to_tensor(sequence: str, device: torch.device = None) -> torch.Tensor:
    """
    Convert amino acid sequence string to one-hot tensor.
    
    Args:
        sequence: amino acid sequence string (e.g., "GLYLEU...")
        device: torch device
        
    Returns:
        tensor: [L, 20] one-hot encoded sequence
    """
    if device is None:
        device = torch.device('cpu')
        
    L = len(sequence)
    tensor = torch.zeros(L, 20, device=device)
    
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            tensor[i, AA_TO_IDX[aa]] = 1.0
        else:
            # Unknown amino acid - uniform distribution
            tensor[i, :] = 1.0 / 20.0
            
    return tensor

def tensor_to_sequence(tensor: torch.Tensor, method: str = 'argmax') -> str:
    """
    Convert sequence prediction tensor to amino acid sequence string.
    
    Args:
        tensor: [L, 20] sequence logits or probabilities
        method: 'argmax', 'sample', or 'threshold'
        
    Returns:
        sequence: amino acid sequence string
    """
    if method == 'argmax':
        indices = torch.argmax(tensor, dim=-1)
        sequence = ''.join([IDX_TO_AA[idx.item()] for idx in indices])
    elif method == 'sample':
        probs = F.softmax(tensor, dim=-1)
        indices = torch.multinomial(probs, 1).squeeze(-1)
        sequence = ''.join([IDX_TO_AA[idx.item()] for idx in indices])
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return sequence

def compute_sequence_accuracy(pred_logits: torch.Tensor, target_sequence: str, mask: torch.Tensor = None) -> float:
    """
    Compute sequence prediction accuracy.
    
    Args:
        pred_logits: [B, L, 20] predicted sequence logits
        target_sequence: target amino acid sequence string
        mask: [B, L] valid residue mask
        
    Returns:
        accuracy: sequence prediction accuracy (0-1)
    """
    # Convert target sequence to tensor
    target_tensor = sequence_to_tensor(target_sequence, device=pred_logits.device)
    target_indices = torch.argmax(target_tensor, dim=-1)  # [L]
    
    # Get predictions
    pred_indices = torch.argmax(pred_logits, dim=-1)  # [B, L]
    
    # Compute accuracy
    if mask is not None:
        valid_mask = mask.bool()
        correct = (pred_indices == target_indices.unsqueeze(0)) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
    else:
        correct = (pred_indices == target_indices.unsqueeze(0))
        accuracy = correct.sum().float() / correct.numel()
        
    return accuracy.item()

def get_aa_3letter(sequence: str) -> list:
    """
    Convert 1-letter amino acid sequence to 3-letter codes.
    
    Args:
        sequence: 1-letter amino acid sequence
        
    Returns:
        list of 3-letter amino acid codes
    """
    aa_1to3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    return [aa_1to3.get(aa, 'ALA') for aa in sequence]
