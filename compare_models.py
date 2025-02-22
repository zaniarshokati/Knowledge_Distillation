# compare_models.py
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from common import TeacherNet, StudentNet, get_data_loaders, evaluate, device

# ==============================
# Utility Functions
# ==============================

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(filepath):
    """Returns the model file size in megabytes (MB)."""
    return os.path.getsize(filepath) / 1e6  # bytes to MB

def measure_inference_time(model, input_size=(1, 3, 32, 32), device=device, num_runs=100):
    """Measures the average inference time per run for the given model."""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    # Warm-up iterations to stabilize performance
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        total_time = time.time() - start_time
    return total_time / num_runs

def get_predictions(model, dataloader):
    """Collects ground truth labels and model predictions for the entire dataset."""
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

# ==============================
# Main Comparison Code
# ==============================

def main():
    # Define paths for saved teacher and student weights
    teacher_path = "teacher_best.pth"
    student_path = "student_best.pth"

    # Check for model files
    if not os.path.exists(teacher_path) or not os.path.exists(student_path):
        raise FileNotFoundError("One or both model files not found. Please ensure 'teacher_best.pth' and 'student_best.pth' exist.")

    # Initialize and load the teacher model
    teacher = TeacherNet(num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device)

    # Initialize and load the student model
    student = StudentNet(num_classes=10)
    student.load_state_dict(torch.load(student_path, map_location=device))
    student.to(device)

    # Get the test dataloader from common.py
    _, testloader = get_data_loaders(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss()

    # ==============================
    # Evaluate Models on Test Set
    # ==============================

    teacher_loss, teacher_acc = evaluate(teacher, testloader, criterion)
    student_loss, student_acc = evaluate(student, testloader, criterion)

    print("=== Performance on Test Set ===")
    print(f"Teacher Test Loss: {teacher_loss:.4f}, Accuracy: {teacher_acc:.4f}")
    print(f"Student Test Loss: {student_loss:.4f}, Accuracy: {student_acc:.4f}")

    # ==============================
    # Compute Model Complexity Metrics
    # ==============================

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    print("\n=== Model Complexity ===")
    print(f"Teacher parameters: {teacher_params}")
    print(f"Student parameters: {student_params}")

    # Save temporary copies to compute disk size
    torch.save(teacher.state_dict(), "teacher_temp.pth")
    torch.save(student.state_dict(), "student_temp.pth")
    teacher_size = get_model_size("teacher_temp.pth")
    student_size = get_model_size("student_temp.pth")
    # Clean up temporary files
    os.remove("teacher_temp.pth")
    os.remove("student_temp.pth")
    print(f"Teacher model file size: {teacher_size:.2f} MB")
    print(f"Student model file size: {student_size:.2f} MB")

    # ==============================
    # Inference Latency Measurement
    # ==============================

    teacher_latency = measure_inference_time(teacher, device=device)
    student_latency = measure_inference_time(student, device=device)
    print("\n=== Inference Latency ===")
    print(f"Teacher inference time: {teacher_latency * 1000:.2f} ms per run")
    print(f"Student inference time: {student_latency * 1000:.2f} ms per run")

    # ==============================
    # Generate Confusion Matrices
    # ==============================

    # Define CIFAR-10 class labels
    cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer",
                      "dog", "frog", "horse", "ship", "truck"]

    # Get predictions for both models
    true_labels, teacher_preds = get_predictions(teacher, testloader)
    _, student_preds = get_predictions(student, testloader)

    # Compute confusion matrices
    from sklearn.metrics import confusion_matrix
    teacher_cm = confusion_matrix(true_labels, teacher_preds)
    student_cm = confusion_matrix(true_labels, student_preds)

    # ==============================
    # Visualization: Confusion Matrices
    # ==============================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(teacher_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cifar10_labels, yticklabels=cifar10_labels)
    plt.title("Teacher Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")

    plt.subplot(1, 2, 2)
    sns.heatmap(student_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cifar10_labels, yticklabels=cifar10_labels)
    plt.title("Student Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()

    # ==============================
    # Visualization: Comparison Bar Chart
    # ==============================
    metrics = {
        "Accuracy": [teacher_acc * 100, student_acc * 100],
        "Test Loss": [teacher_loss, student_loss],
        "Parameters": [teacher_params, student_params],
        "Model Size (MB)": [teacher_size, student_size],
        "Inference Time (ms)": [teacher_latency * 1000, student_latency * 1000]
    }

    labels = list(metrics.keys())
    teacher_vals = [metrics[m][0] for m in labels]
    student_vals = [metrics[m][1] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, teacher_vals, width, label="Teacher", color="tab:blue")
    rects2 = ax.bar(x + width/2, student_vals, width, label="Student", color="tab:green")

    ax.set_ylabel("Value")
    ax.set_title("Comparison of Teacher vs. Student Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset in points
                        textcoords="offset points",
                        ha="center", va="bottom")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()

    # ==============================
    # (Optional) Print Classification Reports
    # ==============================
    from sklearn.metrics import classification_report
    print("\n=== Classification Report for Teacher Model ===")
    print(classification_report(true_labels, teacher_preds, target_names=cifar10_labels))
    print("\n=== Classification Report for Student Model ===")
    print(classification_report(true_labels, student_preds, target_names=cifar10_labels))

if __name__ == '__main__':
    main()
