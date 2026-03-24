# From-BPE-to-Byte-Level-Processing---Building-from-scratch
An educational implementation of a custom Byte-Level BPE (BBPE) tokenizer built from scratch, integrated with an LSTM neural network for next-word prediction

# Subword Tokenization: From BPE to Byte-Level Processing

This repository contains the practical implementation for my university Machine Learning assignment on Advanced Natural Language Processing (NLP) tokenization. 

The project demonstrates the end-to-end pipeline of algorithmic text processing, bypassing black-box libraries to build a custom Byte-Level BPE (BBPE) tokenizer from scratch, which is then integrated with an LSTM neural network for autoregressive next-word prediction.

## 🚀 Features
* **Custom Byte-Level Tokenizer:** A pure Python implementation of the BPE algorithm operating directly on UTF-8 hexadecimal bytes, solving the Out-Of-Vocabulary (OOV) / `<UNK>` token problem.
* **Algorithmic Tracing:** Includes an educational `trace_encoding()` function that prints the step-by-step mathematical merging of bytes into subwords directly to the terminal.
* **LSTM Language Model:** A deep learning architecture (using TensorFlow/Keras) built with Embedding, Dropout, and Long Short-Term Memory layers.
* **Generative Inference:** Autoregressive text generation trained on a subset of Lewis Carroll's *Alice in Wonderland*.

## 🛠️ Technologies Used
* Python 3.x
* TensorFlow / Keras
* NumPy
* Built-in Python `collections`

## ⚙️ How to Run
This code is designed to be fully reproducible and runs flawlessly in Google Colab or any local Jupyter/Python environment. 

1. Clone this repository or copy the `main.py` script.
2. Ensure TensorFlow is installed (`pip install tensorflow numpy`).
3. Run the script. 

The script will automatically:
* Download the required dataset slice dynamically.
* Train the custom BBPE tokenizer (this takes ~1-2 minutes).
* Print a visual trace of the word `'learning'` being tokenized.
* Train the LSTM model on the generated integer tensors.
* Output a generated sentence continuation based on a seed phrase.

## 📊 Expected Output
When running the inference block, the model will output the step-by-step tokenization process, followed by the deep learning training loop. Finally, it will attempt to predict the next logical subwords based on its training:

```text
==================================================
 PREDICTING THE NEXT SENTENCE (INFERENCE) 
==================================================
Input Seed: 'the white rabbit '
AI generated continuation: 'hurriedly ran down the hole '
Full sentence: 'the white rabbit hurriedly ran down the hole '
