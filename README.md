# Knowledge Distillation for Embedded AI

This repository demonstrates **knowledge distillation** to compress deep learning models for embedded AI applications. A high-capacity teacher model is trained on CIFAR-10, and a lightweight student model is trained to mimic it, achieving efficiency with competitive accuracy.

## Features

- **Teacher Model:** Deep CNN trained using standard supervised learning.
- **Student Model:** Compact CNN trained via knowledge distillation.
- **Evaluation & Comparison:** Analyze performance, model size, and inference speed.

## File Structure


knowledge-distillation/
├── common.py           # Shared model definitions, data loaders, and evaluation functions
├── teacher_train.py    # Trains the teacher model
├── student_train.py    # Trains the student model via knowledge distillation
├── compare_models.py   # Evaluates and compares teacher vs. student models
└── README.md           # Project documentation


## Features

