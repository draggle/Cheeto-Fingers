# 🧡 Cheeto-Fingers

**The touch-free mouse interface for when your hands are covered in snack dust.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)

## 🧐 The Problem
Have you ever been eating Cheetos, wings, or pizza while watching YouTube, and suddenly a double-unskippable ad appears? You look at your orange-dusted fingertips. You look at your pristine white mouse. You face a terrible choice: dirty the peripherals, or suffer through the ad.

## 💡 The Solution
**Cheeto-Fingers** uses Computer Vision to track your hand in real-time, allowing you to control your mouse cursor, click, and scroll without ever touching a physical device.

## 🛠️ How It Works (The Tech)
Under the hood, this isn't just magic; it's a pipeline of Deep Learning models:
1.  **Input:** Captures video feed via **OpenCV**.
2.  **Detection:** Uses **Google MediaPipe's** SSD (Single Shot Detector) neural network to locate the hand palm.
3.  **Landmarking:** Runs a secondary model to predict the 3D coordinates of 21 hand knuckles.
4.  **Logic Layer:** Uses vector math (Euclidean distance) to map specific finger pinches to OS-level mouse actions via **PyAutoGUI**.
5.  **Smoothing:** Implements a Jitter Buffer and Exponential Moving Average (EMA) to ensure the cursor is pixel-perfect steady, even if your hand trembles.

## 🎮 Grease-Free Controls
| Action | Gesture |
| :--- | :--- |
| **Move Cursor** | Index Finger Up ☝️ |
| **Left Click** | Pinch Index + Thumb 👌 (Green Dot) |
| **Right Click** | Pinch Ring Finger + Thumb 🤘 (Blue Dot) |
| **Precision Scroll** | Pinch Middle + Thumb & Drag 🤏 (Red Dot) |
| **Quit App** | Press 'q' |

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/draggle/Cheeto-Fingers.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tracking engine: 
   ```bash
   python ai_mouse.py
   ```

## ⚙️ Configuration
You can tune the sensitivity in `ai_mouse.py` to match your snacking style:
```python
scroll_speed = 0.2    # Lower = Heavier/Slower scroll
smoothening = 7       # Higher = Smoother cursor
frame_red = 180       # Green Box size (Smaller = Faster movement)
   
   
