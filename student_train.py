# student_train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import argparse
import os
from tqdm import tqdm
from common import TeacherNet, StudentNet, get_data_loaders, evaluate, device

def train_student(teacher, student, trainloader, testloader, num_epochs, lr, step_size, gamma, T, alpha):
    print("Starting Student Training with Knowledge Distillation...")
    teacher.to(device)
    student.to(device)
    
    # Freeze teacher parameters
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion_ce = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        student.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            # Compute the standard cross-entropy loss
            loss_ce = criterion_ce(student_logits, labels)

            # Compute distillation loss using KL divergence with temperature scaling
            log_student_prob = F.log_softmax(student_logits / T, dim=1)
            teacher_prob = F.softmax(teacher_logits / T, dim=1)
            loss_kd = F.kl_div(log_student_prob, teacher_prob, reduction='batchmean')

            # Combine losses with T^2 scaling on the distillation loss
            loss = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
        scheduler.step()

        # Validate student model
        val_loss, val_acc = evaluate(student, testloader, criterion_ce)
        print(f"Epoch {epoch}/{num_epochs} - Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), "student_best.pth")
            print(f"  --> New best student model saved with accuracy: {best_acc:.4f}")
    print("Student training complete.")
    return student

def main():
    parser = argparse.ArgumentParser(description="Student Model Training with Knowledge Distillation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--step_size", type=int, default=15, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--T", type=float, default=4.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight factor between CE loss and KD loss")
    args = parser.parse_args()

    trainloader, testloader = get_data_loaders(batch_size=args.batch_size)

    # Load the pre-trained teacher model
    teacher_path = "teacher_best.pth"
    if not os.path.exists(teacher_path):
        print("Error: Teacher model not found. Please run teacher_train.py first.")
        return

    teacher = TeacherNet(num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    print("Loaded pre-trained teacher model from teacher_best.pth.")

    student = StudentNet(num_classes=10)
    student = train_student(teacher, student, trainloader, testloader,
                            num_epochs=args.epochs, lr=args.lr,
                            step_size=args.step_size, gamma=args.gamma,
                            T=args.T, alpha=args.alpha)
    
    # Final evaluation of the student model
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(student, testloader, criterion)
    print(f"Final Student Model -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
