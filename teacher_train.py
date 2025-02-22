# teacher_train.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import os
from tqdm import tqdm
from common import TeacherNet, get_data_loaders, evaluate, device

def train_teacher(model, trainloader, testloader, num_epochs, lr, step_size, gamma):
    print("Starting Teacher Training...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # Use tqdm for progress tracking
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
        scheduler.step()

        # Evaluate on validation data
        val_loss, val_acc = evaluate(model, testloader, criterion)
        print(f"Epoch {epoch}/{num_epochs} - Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "teacher_best.pth")
            print(f"  --> New best model saved with accuracy: {best_acc:.4f}")
    print("Teacher training complete.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Teacher Model Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    args = parser.parse_args()

    trainloader, testloader = get_data_loaders(batch_size=args.batch_size)
    teacher = TeacherNet(num_classes=10)

    teacher = train_teacher(teacher, trainloader, testloader,
                            num_epochs=args.epochs, lr=args.lr,
                            step_size=args.step_size, gamma=args.gamma)
    print("Teacher model training complete and saved as teacher_best.pth.")

if __name__ == "__main__":
    main()
