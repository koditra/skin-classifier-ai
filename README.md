# DermVision

DermVision is an AI powered skin condition analysis web app built using PyTorch, Flask, and OpenAI APIs. Users can upload an image, receive an AI prediction for possible skin conditions, view GradCAM visualizations showing what the model focused on, and chat with an AI assistant for simplified explanations.

This project combines:
- Deep learning image classification
- Flask backend APIs
- Frontend UI
- AI chatbot integration
- Explainable AI visualization
- Voice responses using ElevenLabs TTS

---

## Features

- Upload skin images for AI analysis
- Multi class skin condition classification
- GradCAM heatmaps
- AI chat assistant
- Voice output using ElevenLabs
- Mobile friendly UI
- Real time inference with Flask

---

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Flask
- Python

### AI / ML
- PyTorch

### APIs
- OpenAI API
- ElevenLabs API

---

## Model

The model was trained using a custom PyTorch training pipeline on a skin lesion dataset from Kaggle.

Final model accuracy:
- 91.2%

Training pipeline includes:
- Data augmentation
- Image normalization
- Batch training
- GPU/CPU support
- Custom preprocessing pipeline

---

## Dataset

Dataset used:

https://www.kaggle.com/datasets/vinayjayanti/skin-lesion-image-classification/data

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_LINK
cd YOUR_REPO_NAME
```

Create a virtual environment:

### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install packages:

```bash
pip install flask torch torchvision opencv-python pillow numpy requests openai
```

---

## Environment Variables

Set your OpenAI API key:

### Mac/Linux
```bash
export OPENAI_API_KEY="your_api_key"
```

### Windows
```bash
set OPENAI_API_KEY=your_api_key
```

Optional:
Add your ElevenLabs API key if using voice features.

---

## Running The App

Start the Flask server:

```bash
python app.py
```

Then open:

```txt
http://127.0.0.1:5000
```

---

## Pages

- `/` → Image prediction
- `/chat` → AI assistant
- `/gradcam` → GradCAM visualization

---

## Notes

This project is not intended for real medical diagnosis. It is an educational and experimental AI project.

The AI predictions may be incorrect and should not replace professional medical advice.

---

## Future Improvements

- Better UI polish
- Faster streaming responses
- Improved mobile responsiveness
- User accounts
- Cloud deployment
- More accurate model training
- Real time camera support

---

## Credits

Built using:
- PyTorch
- Flask
- OpenAI API
- ElevenLabs
