# Knowledge Distillation for Embedded AI

This repository demonstrates **knowledge distillation** to compress deep learning models for embedded AI applications. A high-capacity teacher model is trained on CIFAR-10, and a lightweight student model is trained to mimic it, achieving efficiency with competitive accuracy.

## Features

- **Teacher Model:** Deep CNN trained using standard supervised learning.
- **Student Model:** Compact CNN trained via knowledge distillation.
- **Evaluation & Comparison:** Analyze performance, model size, and inference speed.

## File Structure

```plaintext
knowledge-distillation/
├── common.py           # Shared model definitions, data loaders, and evaluation functions
├── teacher_train.py    # Trains the teacher model
├── student_train.py    # Trains the student model via knowledge distillation
├── compare_models.py   # Evaluates and compares teacher vs. student models
└── README.md           # Project documentation
How to Use

Train the Teacher Model
python teacher_train.py --epochs 20 --batch_size 128 --lr 0.001
This will save teacher_best.pth.
Train the Student Model
python student_train.py --epochs 30 --batch_size 128 --lr 0.001 --T 4.0 --alpha 0.7
This will save student_best.pth.
Compare & Analyze Models
python compare_models.py
This script outputs accuracy, model size, inference speed, and visualizations.
Example Results

Model	Accuracy	Parameters	Size	Inference Time
Teacher	86.97%	~2.66M	10.65 MB	2.43 ms
Student	82.97%	~0.62M	2.49 MB	0.88 ms
License

This project is open-source under the MIT License.
