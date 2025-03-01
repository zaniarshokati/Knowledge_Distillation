#!/usr/bin/env python3
"""
Teacher Model Training Script

This script trains the teacher model for knowledge distillation.
It implements a standard training loop with validation and model checkpointing.

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from common import TeacherNet, get_data_loaders, evaluate, device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_teacher(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    num_epochs: int,
    lr: float,
    step_size: int,
    gamma: float,
    save_path: str = "teacher_best.pth"
) -> nn.Module:
    """
    Train the teacher model with standard supervised learning.
    
    Args:
        model: The neural network model to train
        trainloader: DataLoader for training data
        testloader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        step_size: Epoch interval for learning rate decay
        gamma: Multiplicative factor for learning rate decay
        save_path: Path to save the best model checkpoint
        
    Returns:
        Trained model
    """
    logger.info("Starting Teacher Model Training...")
    save_path = Path(save_path)
    
    # Move model to device
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Setup optimizer, scheduler and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch {epoch}/{num_epochs}"
        )
        
        for batch_idx, (images, labels) in progress_bar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validation phase
        val_loss, val_acc = evaluate(model, testloader, criterion)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.4f}"
        )
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")
    
    logger.info(f"Teacher training complete. Best accuracy: {best_acc:.4f}")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Teacher Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="LR decay factor"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="teacher_best.pth",
        help="Path to save the best model"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_args()
        logger.info(f"Training with parameters: {args}")
        
        # Get data loaders
        trainloader, testloader = get_data_loaders(batch_size=args.batch_size)
        logger.info(f"Data loaders created with batch size: {args.batch_size}")
        
        # Initialize model
        teacher = TeacherNet(num_classes=10)
        logger.info(f"Initialized TeacherNet with {sum(p.numel() for p in teacher.parameters())} parameters")
        
        # Train model
        teacher = train_teacher(
            teacher,
            trainloader,
            testloader,
            num_epochs=args.epochs,
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
            save_path=args.save_path
        )
        
        logger.info(f"Teacher model training complete and saved as {args.save_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
