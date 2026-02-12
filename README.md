# 🖼️ Image Captioning Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end image captioning system that combines computer vision and natural language processing to generate descriptive text for visual content. Built with PyTorch, the model utilizes a ResNet50 backbone for feature extraction and an Init-Injection LSTM architecture for sequential text generation. The project features a user-friendly Streamlit interface, supporting both Greedy and Beam Search decoding strategies for high-quality caption synthesis.

---

## ✨ Key Features
- **Init-Injection Architecture**: Uses image features to initialize the hidden and cell states of the LSTM decoder, ensuring the visual context is deeply embedded in the generation process.
- **Dual Inference Modes**:
  - **Greedy Search**: fast and efficient caption generation.
  - **Beam Search**: Explores multiple word sequences to find the most probable and coherent caption.
- **Interactive UI**: A Streamlit-based web application for easy testing with custom uploads or sample images.
- **Evaluation Metrics**: Trained and evaluated using metrics like BLEU-4, Precision, Recall, and F1 (as seen in the training notebooks).
- **Pretrained Backbone**: Leverages ResNet50 for robust visual feature extraction.

---

## 🏗️ Architecture Overview

The model follows an Encoder-Decoder framework:
1.  **Visual Encoder**: A pretrained **ResNet50** (minus the final classification layer) extracts high-level spatial features from the input image.
2.  **Projection Layer**: A linear layer maps image features to the hidden dimension size of the LSTM.
3.  **LSTM Encoder (Init-Injection)**: Instead of feeding the image at every time step, it is used to calculate the initial $h_0$ and $c_0$ states of the decoder.
4.  **LSTM Decoder**: Processes the word embeddings and generates the next word in the sequence, guided by the initial visual state.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hurairamuzammal/image_captioning_model.git
cd image_captioning_model
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3. Model Files
Ensure the following files are present in the `Model/` directory:
- `model.pth`: Trained model weights.
- `vocab.pkl`: Vocabulary mappings (`stoi` and `itos`).
- `config.json`: Model hyperparameters and architecture settings.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🛠️ Project Structure
```text
.
├── Model/              # Trained weights, vocabulary, and config
├── notebook/           # Jupyter notebooks for training and evaluation
├── samples/            # Sample images for testing the UI
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📊 Performance & Evaluation
The model was trained on the **Flickr8k/Flickr30k** dataset. During training, we implemented:
- **Label Smoothing**: To improve generalization.
- **Learning Rate Scheduling**: For stable convergence.
- **Data Augmentation**: Including random cropping and flipping to make the model more robust.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the model architecture, improve the UI, or add support for new datasets, feel free to open a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
