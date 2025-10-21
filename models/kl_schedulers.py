#!/usr/bin/env python3
"""
KL Divergence Scheduling Strategies for VAE Training
=====================================================

Implements various KL annealing strategies to prevent posterior collapse
while maintaining reconstruction quality. Based on best practices from:

- Fu et al. "Cyclical Annealing Schedule" (2019)
- Bowman et al. "Generating Sentences from a Continuous Space" (2016)
- Kingma et al. "Improved Variational Inference with Inverse Autoregressive Flow" (2016)

Author: Deep Learning Expert with 15+ years experience
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict


class BaseKLScheduler:
    """Base class for KL weight scheduling."""
    
    def __init__(self, max_weight: float = 1.0):
        """
        Args:
            max_weight: Maximum KL weight to reach
        """
        self.max_weight = max_weight
        self.current_weight = 0.0
        self.history = []
    
    def step(self, epoch: int, total_epochs: int, **kwargs) -> float:
        """
        Update and return KL weight for current epoch.
        
        Args:
            epoch: Current epoch (1-indexed)
            total_epochs: Total training epochs
            **kwargs: Additional metrics (e.g., val_loss, val_rmsd)
            
        Returns:
            Current KL weight
        """
        raise NotImplementedError
    
    def get_state(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'max_weight': self.max_weight,
            'current_weight': self.current_weight,
            'history': self.history
        }
    
    def load_state(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.max_weight = state.get('max_weight', self.max_weight)
        self.current_weight = state.get('current_weight', 0.0)
        self.history = state.get('history', [])


class CyclicalKLScheduler(BaseKLScheduler):
    """
    Cyclical annealing schedule for KL divergence.
    
    Cycles KL weight from 0 to max_weight over multiple cycles.
    This helps the model explore different regions of the latent space
    and prevents posterior collapse.
    
    Reference:
        Fu et al. "Cyclical Annealing Schedule: A Simple Approach to 
        Mitigating KL Vanishing" (2019)
    """
    
    def __init__(self, n_cycles: int = 4, ratio: float = 0.5, 
                 max_weight: float = 1.0, start_weight: float = 0.0):
        """
        Args:
            n_cycles: Number of annealing cycles over training
            ratio: Portion of each cycle spent increasing weight (0.5 = linear, 1.0 = always increasing)
            max_weight: Maximum KL weight at cycle peak
            start_weight: Starting KL weight (usually 0.0)
        """
        super().__init__(max_weight)
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.start_weight = start_weight
        self.current_weight = start_weight
    
    def step(self, epoch: int, total_epochs: int, **kwargs) -> float:
        """
        Compute cyclical KL weight.
        
        The weight follows a sawtooth pattern:
        - Increases linearly from start_weight to max_weight over ratio * cycle_length
        - Either stays at max or decreases back (depending on ratio)
        """
        cycle_length = total_epochs / self.n_cycles
        cycle_position = ((epoch - 1) % cycle_length) / cycle_length
        
        if cycle_position < self.ratio:
            # Increasing phase
            progress = cycle_position / self.ratio
            self.current_weight = self.start_weight + (self.max_weight - self.start_weight) * progress
        else:
            # Hold at maximum (or decrease if ratio < 1.0)
            if self.ratio < 1.0:
                # Decreasing phase
                progress = (cycle_position - self.ratio) / (1.0 - self.ratio)
                self.current_weight = self.max_weight - (self.max_weight - self.start_weight) * progress
            else:
                self.current_weight = self.max_weight
        
        self.history.append(self.current_weight)
        return self.current_weight
    
    def __repr__(self):
        return (f"CyclicalKLScheduler(n_cycles={self.n_cycles}, ratio={self.ratio}, "
                f"max_weight={self.max_weight}, current={self.current_weight:.4f})")


class MonotonicKLScheduler(BaseKLScheduler):
    """
    Standard monotonic β-VAE warmup schedule.
    
    Linearly increases KL weight from 0 to max over warmup_epochs,
    then holds constant. This is the classic β-VAE approach.
    
    Reference:
        Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a 
        Constrained Variational Framework" (2017)
    """
    
    def __init__(self, warmup_epochs: int = 50, max_weight: float = 1.0,
                 hold_epochs: Optional[int] = None):
        """
        Args:
            warmup_epochs: Number of epochs to warm up KL weight
            max_weight: Target KL weight after warmup
            hold_epochs: Optional epochs to hold at intermediate weight before final max
        """
        super().__init__(max_weight)
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.intermediate_weight = max_weight * 0.5 if hold_epochs else max_weight
    
    def step(self, epoch: int, total_epochs: int, **kwargs) -> float:
        """Linear warmup then constant."""
        if epoch <= self.warmup_epochs:
            # Linear warmup
            self.current_weight = self.max_weight * (epoch / self.warmup_epochs)
        elif self.hold_epochs and epoch <= self.warmup_epochs + self.hold_epochs:
            # Hold at intermediate
            self.current_weight = self.intermediate_weight
        else:
            # Final weight
            self.current_weight = self.max_weight
        
        self.history.append(self.current_weight)
        return self.current_weight
    
    def __repr__(self):
        return (f"MonotonicKLScheduler(warmup_epochs={self.warmup_epochs}, "
                f"max_weight={self.max_weight}, current={self.current_weight:.4f})")


class AdaptiveKLScheduler(BaseKLScheduler):
    """
    Adaptive KL scheduling based on reconstruction quality.
    
    Increases KL when reconstruction is good (RMSD < target),
    decreases when reconstruction is poor. This helps balance
    reconstruction vs. latent space learning dynamically.
    
    Note: Requires validation metrics to be passed via step(**kwargs)
    """
    
    def __init__(self, target_rmsd: float = 1.5, min_weight: float = 0.1,
                 max_weight: float = 10.0, adapt_rate: float = 0.05,
                 warmup_epochs: int = 20):
        """
        Args:
            target_rmsd: Target reconstruction RMSD (Angstroms)
            min_weight: Minimum KL weight
            max_weight: Maximum KL weight
            adapt_rate: Rate of adaptation (0.05 = 5% change per epoch)
            warmup_epochs: Initial warmup before adaptation
        """
        super().__init__(max_weight)
        self.target_rmsd = target_rmsd
        self.min_weight = min_weight
        self.adapt_rate = adapt_rate
        self.warmup_epochs = warmup_epochs
        self.current_weight = min_weight
    
    def step(self, epoch: int, total_epochs: int, val_rmsd: Optional[float] = None,
             **kwargs) -> float:
        """
        Adapt weight based on validation RMSD.
        
        Args:
            val_rmsd: Validation RMSD in Angstroms
        """
        if epoch <= self.warmup_epochs:
            # Initial warmup
            self.current_weight = self.min_weight + (self.max_weight - self.min_weight) * (epoch / self.warmup_epochs) * 0.5
        elif val_rmsd is not None:
            # Adapt based on reconstruction quality
            if val_rmsd < self.target_rmsd:
                # RMSD is good, can afford more KL regularization
                self.current_weight *= (1 + self.adapt_rate)
            else:
                # RMSD is poor, reduce KL to improve reconstruction
                self.current_weight *= (1 - self.adapt_rate)
            
            # Clip to bounds
            self.current_weight = np.clip(self.current_weight, self.min_weight, self.max_weight)
        
        self.history.append(self.current_weight)
        return self.current_weight
    
    def __repr__(self):
        return (f"AdaptiveKLScheduler(target_rmsd={self.target_rmsd:.2f}, "
                f"range=[{self.min_weight:.2f}, {self.max_weight:.2f}], "
                f"current={self.current_weight:.4f})")


class ExponentialKLScheduler(BaseKLScheduler):
    """
    Exponential warmup schedule.
    
    Useful for very deep VAEs where gradual warmup is critical.
    Weight increases exponentially rather than linearly.
    """
    
    def __init__(self, warmup_epochs: int = 50, max_weight: float = 1.0,
                 steepness: float = 2.0):
        """
        Args:
            warmup_epochs: Number of epochs for warmup
            max_weight: Final KL weight
            steepness: Controls exponential curve (higher = steeper)
        """
        super().__init__(max_weight)
        self.warmup_epochs = warmup_epochs
        self.steepness = steepness
    
    def step(self, epoch: int, total_epochs: int, **kwargs) -> float:
        """Exponential warmup."""
        if epoch <= self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            # Exponential curve: 0 -> 1
            exponential_progress = (np.exp(self.steepness * progress) - 1) / (np.exp(self.steepness) - 1)
            self.current_weight = self.max_weight * exponential_progress
        else:
            self.current_weight = self.max_weight
        
        self.history.append(self.current_weight)
        return self.current_weight
    
    def __repr__(self):
        return (f"ExponentialKLScheduler(warmup_epochs={self.warmup_epochs}, "
                f"max_weight={self.max_weight}, current={self.current_weight:.4f})")


class FreeBitsKLLoss(nn.Module):
    """
    KL loss with free bits constraint.
    
    Ensures a minimum KL divergence per latent dimension to prevent
    over-regularization and posterior collapse. This allows the model
    to use the latent space effectively.
    
    Reference:
        Kingma et al. "Improved Variational Inference with Inverse 
        Autoregressive Flow" (2016)
    """
    
    def __init__(self, free_bits: float = 2.0, min_kl: float = 0.0,
                 use_free_bits: bool = True):
        """
        Args:
            free_bits: Minimum KL per dimension (in nats)
            min_kl: Hard minimum KL value
            use_free_bits: Whether to apply free bits (for ablations)
        """
        super().__init__()
        self.free_bits = free_bits
        self.min_kl = min_kl
        self.use_free_bits = use_free_bits
    
    def forward(self, mu: torch.Tensor, lv: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                reduce: str = 'mean') -> torch.Tensor:
        """
        Compute KL divergence with free bits.
        
        Args:
            mu: Mean of q(z|x), shape [..., latent_dim]
            lv: Log variance of q(z|x), shape [..., latent_dim]
            mask: Optional mask for local latents, shape [...]
            reduce: 'mean', 'sum', or 'none'
            
        Returns:
            KL divergence loss (scalar if reduce != 'none')
        """
        # Standard KL divergence: KL(q(z|x) || p(z))
        # For diagonal Gaussian: 0.5 * (exp(lv) + mu^2 - 1 - lv)
        kl = 0.5 * (lv.exp() + mu.pow(2) - 1.0 - lv)
        
        # Apply free bits constraint
        if self.use_free_bits:
            kl = torch.max(kl, torch.tensor(self.free_bits, device=kl.device))
        
        # Apply hard minimum
        if self.min_kl > 0:
            kl = torch.max(kl, torch.tensor(self.min_kl, device=kl.device))
        
        # Sum over latent dimensions
        kl = kl.sum(dim=-1)  # [...] 
        
        # Apply mask if provided (for local latents)
        if mask is not None:
            kl = kl * mask
            if reduce == 'mean':
                return kl.sum() / mask.sum().clamp(min=1.0)
            elif reduce == 'sum':
                return kl.sum()
        
        # Reduce
        if reduce == 'mean':
            return kl.mean()
        elif reduce == 'sum':
            return kl.sum()
        else:
            return kl


def create_kl_scheduler(schedule_type: str, max_weight: float = 1.0,
                       warmup_epochs: int = 50, n_cycles: int = 4,
                       **kwargs) -> BaseKLScheduler:
    """
    Factory function to create KL scheduler.
    
    Args:
        schedule_type: One of ['cyclical', 'monotonic', 'adaptive', 'exponential']
        max_weight: Maximum KL weight
        warmup_epochs: Warmup epochs (for monotonic/exponential)
        n_cycles: Number of cycles (for cyclical)
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        KL scheduler instance
    """
    schedule_type = schedule_type.lower()
    
    if schedule_type == 'cyclical':
        ratio = kwargs.get('ratio', 0.5)
        return CyclicalKLScheduler(n_cycles=n_cycles, ratio=ratio, max_weight=max_weight)
    
    elif schedule_type == 'monotonic':
        return MonotonicKLScheduler(warmup_epochs=warmup_epochs, max_weight=max_weight)
    
    elif schedule_type == 'adaptive':
        target_rmsd = kwargs.get('target_rmsd', 1.5)
        min_weight = kwargs.get('min_weight', 0.1)
        adapt_rate = kwargs.get('adapt_rate', 0.05)
        return AdaptiveKLScheduler(
            target_rmsd=target_rmsd, min_weight=min_weight,
            max_weight=max_weight, adapt_rate=adapt_rate, warmup_epochs=warmup_epochs
        )
    
    elif schedule_type == 'exponential':
        steepness = kwargs.get('steepness', 2.0)
        return ExponentialKLScheduler(
            warmup_epochs=warmup_epochs, max_weight=max_weight, steepness=steepness
        )
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. "
                        f"Choose from ['cyclical', 'monotonic', 'adaptive', 'exponential']")


# Convenience functions for backward compatibility
def get_cyclical_kl_weight(epoch: int, total_epochs: int, n_cycles: int = 4,
                           ratio: float = 0.5, max_weight: float = 1.0) -> float:
    """Legacy function for cyclical KL weight."""
    scheduler = CyclicalKLScheduler(n_cycles=n_cycles, ratio=ratio, max_weight=max_weight)
    return scheduler.step(epoch, total_epochs)


def get_monotonic_kl_weight(epoch: int, warmup_epochs: int = 50,
                            max_weight: float = 1.0) -> float:
    """Legacy function for monotonic KL weight."""
    scheduler = MonotonicKLScheduler(warmup_epochs=warmup_epochs, max_weight=max_weight)
    return scheduler.step(epoch, warmup_epochs)


if __name__ == "__main__":
    """Visualization of different KL schedules."""
    import matplotlib.pyplot as plt
    
    total_epochs = 200
    epochs = range(1, total_epochs + 1)
    
    # Test different schedulers
    schedulers = {
        'Cyclical (4 cycles)': CyclicalKLScheduler(n_cycles=4, ratio=0.5, max_weight=2.0),
        'Monotonic': MonotonicKLScheduler(warmup_epochs=40, max_weight=2.0),
        'Exponential': ExponentialKLScheduler(warmup_epochs=40, max_weight=2.0, steepness=3.0),
        'Cyclical (8 cycles, fast)': CyclicalKLScheduler(n_cycles=8, ratio=0.3, max_weight=2.0),
    }
    
    plt.figure(figsize=(14, 8))
    
    for name, scheduler in schedulers.items():
        weights = [scheduler.step(e, total_epochs) for e in epochs]
        plt.plot(epochs, weights, linewidth=2, label=name, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('KL Weight', fontsize=14)
    plt.title('KL Divergence Annealing Schedules', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('kl_schedules_comparison.png', dpi=200)
    print("✅ Saved schedule visualization to kl_schedules_comparison.png")
    
    # Print statistics
    print("\nScheduler Statistics:")
    print("=" * 60)
    for name, scheduler in schedulers.items():
        weights = scheduler.history
        print(f"\n{name}:")
        print(f"  Mean weight: {np.mean(weights):.3f}")
        print(f"  Max weight:  {np.max(weights):.3f}")
        print(f"  Min weight:  {np.min(weights):.3f}")
        print(f"  Std dev:     {np.std(weights):.3f}")

