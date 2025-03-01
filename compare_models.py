#!/usr/bin/env python3
"""
Model Comparison Tool for Knowledge Distillation

This script compares teacher and student models trained through knowledge distillation,
analyzing their performance, complexity, and efficiency metrics. It generates detailed
visualizations and reports to help understand the trade-offs between the models.

Author: [Your Name]
Date: [Current Date]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.utils.data import DataLoader
from common import TeacherNet, StudentNet, get_data_loaders, evaluate, device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases
ModelType = Union[TeacherNet, StudentNet]
NumericType = Union[int, float]

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    loss: float
    accuracy: float
    parameters: int
    size_mb: float
    inference_time_ms: float

class ModelComparison:
    """Handles comparison between teacher and student models."""
    
    def __init__(
        self,
        teacher_path: str = "teacher_best.pth",
        student_path: str = "student_best.pth",
        batch_size: int = 128
    ):
        self.teacher_path = Path(teacher_path)
        self.student_path = Path(student_path)
        self.batch_size = batch_size
        self.cifar10_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        self.teacher: Optional[TeacherNet] = None
        self.student: Optional[StudentNet] = None
        self.testloader: Optional[DataLoader] = None

    def setup(self) -> None:
        """Initialize models and data loaders."""
        try:
            self._validate_model_files()
            self._load_models()
            _, self.testloader = get_data_loaders(batch_size=self.batch_size)
            logger.info("Setup completed successfully")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

    def _validate_model_files(self) -> None:
        """Ensure model files exist."""
        if not self.teacher_path.exists() or not self.student_path.exists():
            raise FileNotFoundError(
                "Model files not found. Ensure both 'teacher_best.pth' and 'student_best.pth' exist."
            )

    def _load_models(self) -> None:
        """Load teacher and student models."""
        self.teacher = TeacherNet(num_classes=10)
        self.student = StudentNet(num_classes=10)
        
        self.teacher.load_state_dict(
            torch.load(self.teacher_path, map_location=device)
        )
        self.student.load_state_dict(
            torch.load(self.student_path, map_location=device)
        )
        
        self.teacher.to(device)
        self.student.to(device)

    @staticmethod
    def count_parameters(model: ModelType) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_size(filepath: Path) -> float:
        """Get model file size in MB."""
        return filepath.stat().st_size / 1e6

    @staticmethod
    def measure_inference_time(
        model: ModelType,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
        num_runs: int = 100
    ) -> float:
        """Measure average inference time per run."""
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            # Warm-up runs
            for _ in range(10):
                _ = model(dummy_input)
            
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
            total_time = time.time() - start_time
            
        return total_time / num_runs

    def get_predictions(
        self,
        model: ModelType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on test set."""
        if not self.testloader:
            raise RuntimeError("TestLoader not initialized. Call setup() first.")
            
        model.eval()
        all_labels, all_preds = [], []
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        return np.array(all_labels), np.array(all_preds)

    def compute_metrics(self) -> Tuple[ModelMetrics, ModelMetrics]:
        """Compute comprehensive metrics for both models."""
        if not all([self.teacher, self.student, self.testloader]):
            raise RuntimeError("Models or TestLoader not initialized. Call setup() first.")

        criterion = nn.CrossEntropyLoss()
        
        # Teacher metrics
        teacher_loss, teacher_acc = evaluate(self.teacher, self.testloader, criterion)
        teacher_params = self.count_parameters(self.teacher)
        torch.save(self.teacher.state_dict(), "teacher_temp.pth")
        teacher_size = self.get_model_size(Path("teacher_temp.pth"))
        teacher_latency = self.measure_inference_time(self.teacher)
        Path("teacher_temp.pth").unlink()
        
        # Student metrics
        student_loss, student_acc = evaluate(self.student, self.testloader, criterion)
        student_params = self.count_parameters(self.student)
        torch.save(self.student.state_dict(), "student_temp.pth")
        student_size = self.get_model_size(Path("student_temp.pth"))
        student_latency = self.measure_inference_time(self.student)
        Path("student_temp.pth").unlink()
        
        teacher_metrics = ModelMetrics(
            loss=teacher_loss,
            accuracy=teacher_acc,
            parameters=teacher_params,
            size_mb=teacher_size,
            inference_time_ms=teacher_latency * 1000
        )
        
        student_metrics = ModelMetrics(
            loss=student_loss,
            accuracy=student_acc,
            parameters=student_params,
            size_mb=student_size,
            inference_time_ms=student_latency * 1000
        )
        
        return teacher_metrics, student_metrics

    def plot_confusion_matrices(self) -> None:
        """Plot confusion matrices for both models."""
        if not all([self.teacher, self.student]):
            raise RuntimeError("Models not initialized. Call setup() first.")
            
        true_labels, teacher_preds = self.get_predictions(self.teacher)
        _, student_preds = self.get_predictions(self.student)
        
        teacher_cm = confusion_matrix(true_labels, teacher_preds)
        student_cm = confusion_matrix(true_labels, student_preds)
        
        plt.figure(figsize=(14, 6))
        
        for idx, (cm, title) in enumerate([
            (teacher_cm, "Teacher Model"),
            (student_cm, "Student Model")
        ], 1):
            plt.subplot(1, 2, idx)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.cifar10_labels,
                yticklabels=self.cifar10_labels
            )
            plt.title(f"{title} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True Label")
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrices saved as 'confusion_matrices.png'")

    def plot_comparison_metrics(
        self,
        teacher_metrics: ModelMetrics,
        student_metrics: ModelMetrics
    ) -> None:
        """Plot comparison bar chart of model metrics."""
        metrics = {
            "Accuracy (%)": [teacher_metrics.accuracy * 100, student_metrics.accuracy * 100],
            "Loss": [teacher_metrics.loss, student_metrics.loss],
            "Parameters": [teacher_metrics.parameters, student_metrics.parameters],
            "Size (MB)": [teacher_metrics.size_mb, student_metrics.size_mb],
            "Inference (ms)": [teacher_metrics.inference_time_ms, student_metrics.inference_time_ms]
        }
        
        labels = list(metrics.keys())
        teacher_vals = [metrics[m][0] for m in labels]
        student_vals = [metrics[m][1] for m in labels]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, teacher_vals, width, label="Teacher", color="tab:blue")
        rects2 = ax.bar(x + width/2, student_vals, width, label="Student", color="tab:green")
        
        ax.set_ylabel("Value")
        ax.set_title("Model Comparison Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom"
                )
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Comparison metrics plot saved as 'model_comparison.png'")

    def print_classification_reports(self) -> None:
        """Generate and print classification reports for both models."""
        if not all([self.teacher, self.student]):
            raise RuntimeError("Models not initialized. Call setup() first.")
            
        true_labels, teacher_preds = self.get_predictions(self.teacher)
        _, student_preds = self.get_predictions(self.student)
        
        logger.info("\n=== Classification Report for Teacher Model ===")
        print(classification_report(true_labels, teacher_preds,
                                 target_names=self.cifar10_labels))
        
        logger.info("\n=== Classification Report for Student Model ===")
        print(classification_report(true_labels, student_preds,
                                 target_names=self.cifar10_labels))

def main() -> None:
    """Main execution function."""
    try:
        comparison = ModelComparison()
        comparison.setup()
        
        teacher_metrics, student_metrics = comparison.compute_metrics()
        
        # Log basic metrics
        logger.info("=== Performance Metrics ===")
        logger.info(f"Teacher - Loss: {teacher_metrics.loss:.4f}, Accuracy: {teacher_metrics.accuracy:.4f}")
        logger.info(f"Student - Loss: {student_metrics.loss:.4f}, Accuracy: {student_metrics.accuracy:.4f}")
        
        # Generate visualizations
        comparison.plot_confusion_matrices()
        comparison.plot_comparison_metrics(teacher_metrics, student_metrics)
        comparison.print_classification_reports()
        
        logger.info("Model comparison completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
