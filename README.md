# 🎥 Real-Time Color Detection using OpenCV (Python)

This project demonstrates **real-time color detection** using a webcam and **OpenCV**.  
It detects a **specific color (Green by default)** in the live video feed and draws a **bounding box** around the detected object.

This project is beginner-friendly and great for learning:
- OpenCV basics
- Color spaces (BGR → HSV)
- Image masking
- Bounding box detection

---

## 📌 Features

- 📷 Real-time webcam video capture  
- 🎨 HSV color space for accurate color detection  
- 🟩 Bounding box around detected color  
- 🔁 Handles HSV color wrap-around (important for Red)  
- 🧠 Simple and clean logic for beginners  

---

## 🧠 Why HSV Instead of RGB?

HSV (Hue, Saturation, Value) is better for color detection because:
- Hue represents the actual color
- Less affected by lighting changes
- Easier to define color ranges

---

## 📂 Project Structure
color-detection/
│
├── main.py # Main script (video capture & detection)
├── util.py # Helper function to get HSV limits
├── requirements.txt # Python dependencies
├── README.md # Documentation


## ⚙️ Installation


### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2️⃣ (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 📦 Usage
Run the main script:
```bash
python main.py
