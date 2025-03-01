#!/usr/bin/env python3
"""
Common Utilities for Knowledge Distillation

This module provides shared components for the knowledge distillation project,
including model architectures, data loading utilities, and evaluation functions.

Author: [Your Name]
Date: [Current Date]
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model Definitions
# -------------------------
class TeacherNet(nn.Module):
    """
    Teacher network with a larger architecture.
    
    A CNN model with 3 convolutional blocks and a classifier head,
    designed to achieve high accuracy on CIFAR-10.
    """
    
    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize the teacher network.
        
        Args:
            num_classes: Number of output classes
        """
        super(TeacherNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 16x16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 8x8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # output: 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
            
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


class StudentNet(nn.Module):
    """
    Student network with a smaller architecture.
    
    A lightweight CNN model with 3 convolutional blocks and a classifier head,
    designed to be more efficient while maintaining reasonable accuracy.
    """
    
    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize the student network.
        
        Args:
            num_classes: Number of output classes
        """
        super(StudentNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # output: 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
            
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# -------------------------
# Data Loading and Evaluation
# -------------------------
def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for CIFAR-10 training and testing.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        data_root: Directory to store the dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # Create datasets
    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CIFAR-10 dataset: {str(e)}")

    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return trainloader, testloader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Neural network model to evaluate
        dataloader: DataLoader containing evaluation data
        criterion: Loss function
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate metrics
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy
