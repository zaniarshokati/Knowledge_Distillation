#!/usr/bin/env python3
"""
Student Model Training Script with Knowledge Distillation

This script trains the student model using knowledge distillation from a pre-trained teacher model.
It implements the distillation loss combining standard cross-entropy with KL divergence.

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
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from common import TeacherNet, StudentNet, get_data_loaders, evaluate, device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_student(
    teacher: nn.Module,
    student: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    num_epochs: int,
    lr: float,
    step_size: int,
    gamma: float,
    T: float,
    alpha: float,
    save_path: str = "student_best.pth"
) -> nn.Module:
    """
    Train the student model with knowledge distillation from the teacher.
    
    Args:
        teacher: Pre-trained teacher model
        student: Student model to train
        trainloader: DataLoader for training data
        testloader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        step_size: Epoch interval for learning rate decay
        gamma: Multiplicative factor for learning rate decay
        T: Temperature parameter for softening probability distributions
        alpha: Weight balancing cross-entropy and distillation loss
        save_path: Path to save the best model checkpoint
        
    Returns:
        Trained student model
    """
    logger.info("Starting Student Training with Knowledge Distillation...")
    save_path = Path(save_path)
    
    # Move models to device
    teacher.to(device)
    student.to(device)
    logger.info(f"Using device: {device}")
    
    # Freeze teacher parameters
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logger.info("Teacher model frozen for distillation")

    # Setup optimizer, scheduler and loss function
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion_ce = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Training phase
        student.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0

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

            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(images)
                
            # Get student predictions
            student_logits = student(images)

            # Compute the standard cross-entropy loss
            loss_ce = criterion_ce(student_logits, labels)
            running_ce_loss += loss_ce.item()

            # Compute distillation loss using KL divergence with temperature scaling
            log_student_prob = F.log_softmax(student_logits / T, dim=1)
            teacher_prob = F.softmax(teacher_logits / T, dim=1)
            loss_kd = F.kl_div(log_student_prob, teacher_prob, reduction='batchmean')
            running_kd_loss += loss_kd.item()

            # Combine losses with T^2 scaling on the distillation loss
            loss = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kd

            # Backpropagation
            optimizer.zero_grad()
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
        val_loss, val_acc = evaluate(student, testloader, criterion_ce)
        
        # Log metrics
        avg_ce_loss = running_ce_loss / len(trainloader)
        avg_kd_loss = running_kd_loss / len(trainloader)
        
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {avg_loss:.4f} "
            f"(CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f}), "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), save_path)
            logger.info(f"New best student model saved with accuracy: {best_acc:.4f}")
    
    logger.info(f"Student training complete. Best accuracy: {best_acc:.4f}")
    return student


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Student Model Training with Knowledge Distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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
        default=15,
        help="Step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="LR decay factor"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=4.0,
        help="Temperature for distillation"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight factor between CE loss and KD loss"
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="teacher_best.pth",
        help="Path to the teacher model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="student_best.pth",
        help="Path to save the best student model"
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

        # Load the pre-trained teacher model
        teacher_path = Path(args.teacher_path)
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher model not found at {teacher_path}. "
                "Please run teacher_train.py first."
            )

        # Initialize models
        teacher = TeacherNet(num_classes=10)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        logger.info(f"Loaded pre-trained teacher model from {teacher_path}")

        student = StudentNet(num_classes=10)
        logger.info(f"Initialized StudentNet with {sum(p.numel() for p in student.parameters())} parameters")
        
        # Train student model
        student = train_student(
            teacher,
            student,
            trainloader,
            testloader,
            num_epochs=args.epochs,
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
            T=args.T,
            alpha=args.alpha,
            save_path=args.save_path
        )
        
        # Final evaluation
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(student, testloader, criterion)
        logger.info(f"Final Student Model -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Calculate compression metrics
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        compression_ratio = teacher_params / student_params
        
        logger.info(f"Model compression: {compression_ratio:.2f}x reduction in parameters")
        logger.info(f"Teacher parameters: {teacher_params:,}")
        logger.info(f"Student parameters: {student_params:,}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
