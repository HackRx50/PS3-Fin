# Image Forgery Detection Using Vision Transformer

![Model Architecture](https://github.com/user-attachments/assets/90a6dcf3-5d03-462a-917a-94dd263cd815)

This repository implements a model for detecting image forgery using Vision Transformers. It leverages advanced deep learning techniques for robust and accurate detection of manipulated images, ensuring high reliability in image authenticity verification.

## Features

- **Vision Transformer (ViT)**: Utilized for extracting deep features from images for forgery detection.
- **Robust Training Pipeline**: Includes data preprocessing, augmentation, and model training scripts.
- **Evaluation Metrics**: Implements detailed evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Scalability**: Designed to work with large datasets and diverse image manipulation techniques.

## Repository Structure

```
Text-Forgery-Detection-Vision-Transformer/
│
├── model/                      # Contains model architecture and related scripts
│   ├── __init__.py
│   ├── dtd.py                  # Vision Transformer definition
│   ├── eval_dtd.py             # Evaluation scripts
│   ├── fph.py                  # Utility functions for model training
│   └── swins.py                # Swin Transformer utilities
│
├── environment.yml             # Conda environment setup file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/Text-Forgery-Detection-Vision-Transformer.git
   cd Text-Forgery-Detection-Vision-Transformer
   ```

2. **Set up the environment**:

   Using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate forgery-detection
   ```

   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model**:
   - Prepare your dataset by placing it in the `data/` directory.
   - Run the training script:
     ```bash
     python train.py --data_dir ./data --epochs 50 --batch_size 32
     ```

2. **Evaluate the Model**:
   ```bash
   python eval_dtd.py --model_path ./checkpoints/model.pth --test_dir ./data/test
   ```

3. **Detect Forgery in Images**:
   Use the inference script to check a single image:
   ```bash
   python infer.py --image_path ./sample.jpg --model_path ./checkpoints/model.pth
   ```

## Dataset

Ensure your dataset contains labeled images organized into appropriate folders for training and testing. Example structure:

```
data/
├── train/
│   ├── real/
│   └── forged/
├── test/
│   ├── real/
│   └── forged/
```

## Results

<>

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
