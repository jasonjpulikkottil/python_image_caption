# Image Caption App

This is a **Streamlit app** that allows users to upload an image (**PNG, JPEG, JPG**) and generates an explanation for it in **English** using the **BLIP model** from Hugging Face.


## 🚀 Features
✅ Upload images (PNG, JPEG, JPG)  
✅ AI-generated caption using **BLIP**  
✅ Simple and interactive UI  
✅ Works **offline** (no API required)  
✅ Supports CPU (can be switched to GPU)


## 📦 Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/image-explanation-app.git
   cd image-explanation-app
   ```

2. **Create a virtual environment (optional but recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```


## ▶️ Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open the browser and go to the **local URL** shown in the terminal.
3. Upload an image and get its explanation in English!


## 📜 Requirements

Make sure you have the following dependencies installed:

```
streamlit
Pillow
torch
transformers
```

## 🛠️ How It Works

- The app loads the **BLIP model** (`Salesforce/blip-image-captioning-base`).
- Users can upload an image.
- The model processes the image and generates a textual explanation.
- The explanation is displayed on the Streamlit UI.

## ⚡ Model Details

- **BLIP (Bootstrapped Language-Image Pretraining)** is an image captioning model by Salesforce.
- The model generates descriptions by analyzing the image content.
- More details: [Hugging Face Model](https://huggingface.co/Salesforce/blip-image-captioning-base)


## 🖥️ GPU Acceleration (Optional)

If you have a **CUDA-compatible GPU**, modify `app.py` to use GPU:
```python
inputs = processor(image, return_tensors="pt").to("cuda")  # Change "cpu" to "cuda"
model.to("cuda")
```


## 📌 Future Improvements

- 🌍 Support for **multiple languages** (e.g., Malayalam)
- 🎨 Better UI design
- ⚡ Faster inference with optimized models


## 🤝 Contributing

Contributions are welcome! Feel free to fork, open issues, or submit pull requests.


## 📜 License

This project is **open-source** and available under the **MIT License**.


