"""Implements knowledge distillation from DDM ensemble to single model (Paper Section 3.6)"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import math

class DiffusionDistiller:
    def __init__(self, teacher, student, num_train_timesteps=1000, 
                 loss_fn=nn.MSELoss(), lr=1e-4, warmup_ratio=0.05):
        """
        Args:
            teacher: DecentralizedFlowMatcher instance with router and experts
            student: ExpertDiT model to distill into
            num_train_timesteps: Total diffusion steps
            loss_fn: Loss between teacher and student predictions
            lr: Base learning rate
            warmup_ratio: Warmup period as ratio of total steps
        """
        self.teacher = teacher
        self.student = student
        self.loss_fn = loss_fn
        self.num_timesteps = num_train_timesteps
        
        # Paper-matched optimization setup
        self.optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.1)
        self.warmup_steps = int(warmup_ratio * num_train_timesteps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_lambda
        )
        
    def _get_lr_lambda(self, step):
        # Paper's learning rate schedule (warmup + cosine decay)
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.num_timesteps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(self, x0):
        """
        Single distillation step following paper Algorithm 3
        Args:
            x0: Clean training samples from dataset
        """
        # Random timestep and noise
        t = torch.randint(0, self.num_timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        
        # Diffuse samples
        alpha_bar = self.teacher.alpha_bar.to(x0.device)
        x_t = self.teacher.forward_diffuse(x0, t, alpha_bar, noise)
        
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_pred = self._get_teacher_flow(x_t, t)
        
        # Student prediction
        student_pred = self.student(x_t, t)
        
        # Compute distillation loss
        loss = self.loss_fn(student_pred, teacher_pred)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return loss.item()

    def _get_teacher_flow(self, x_t, t):
        """Get combined expert predictions using router"""
        # Get router probabilities
        logits = self.teacher.router(x_t, t/self.num_timesteps)
        probs = torch.softmax(logits, dim=-1)
        
        # Top-1 expert selection (paper recommendation)
        _, top_indices = torch.topk(probs, k=1, dim=-1)
        
        # Combine expert predictions
        combined = torch.zeros_like(x_t)
        for expert_idx, expert in enumerate(self.teacher.experts):
            mask = (top_indices == expert_idx).squeeze()
            if mask.any():
                combined[mask] = expert(x_t[mask], t[mask])
        return combined

    def train(self, loader, epochs=1):
        """Full distillation training loop"""
        self.student.train()
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(loader, desc=f"Distilling [Epoch {epoch+1}/{epochs}]")
            for batch in pbar:
                loss = self.train_step(batch.to(self.student.device))
                total_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            print(f"Epoch {epoch+1} Average Loss: {total_loss/len(loader):.4f}")
        return self.student
