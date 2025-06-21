# 🚘 SARATHI – Unified Attention Monitoring for Transportation

> "A moment’s lapse. A lifetime lost. SARATHI watches, warns, and saves."

---

## 📍 Overview

**SARATHI** is a real-time AI-based driver and pilot attention monitoring system that **detects drowsiness**, **unsafe posture**, and **inattentiveness** using only a single webcam. It serves as a **digital co-pilot**, ensuring that no fatigue-related accident goes unnoticed.

Developed during **Hack4Bengal 4.0** – Eastern India's Largest Hackathon, SARATHI addresses one of the deadliest causes of transport-related deaths — **fatigue**.

---
| Front View | Side View |
|--------|---------|
| ![Front View](https://github.com/user-attachments/assets/8fb8087d-81d5-4952-8b27-e60a220ff531) | ![Side View](https://github.com/user-attachments/assets/3d908436-dbb5-400b-803e-ace2f6a058f7) |



## 🚀 Key Features (USPs)

- ✅ **Dual-layer Detection**: Simultaneously monitors **eye behavior**, **head pose**, and **body posture***  
- 🖼 **Screenshot Capture**: Takes periodic screenshots during alert states  
- 📣 **Smart Alerts**: Audio alarm + visual warnings when fatigue is detected  
- 💡 **Lightweight**: No heavy ML models — optimized for laptops & Raspberry Pi  
- 🔋 **Low Resource Usage**: CPU-efficient, real-time performance  
- 🔧 **Fully Modular**: Easily extendable for dual camera or IoT deployment

---

## 📊 Impact & Benefits

| 🚗 Road Safety | 🚌 Fleet Monitoring | ✈ Pilot Awareness |
|----------------|---------------------|--------------------|
| Reduces fatigue-related accidents | Real-time monitoring for public transport | Can scale to cockpit fatigue detection |

---

## 🧠 Tech Stack

| Category | Tools/Technologies |
|---------|-------------------|
| Language | Python |
| Libraries | OpenCV, MediaPipe, NumPy, Pygame |
| UI | Streamlit |
| Audio | Pygame |
| Utilities | jsonschema, tornado |
| Future Hardware | Raspberry Pi 4, external buzzer/cam |

---

## ⚙ Installation & Run Guide

### 🔄 1. Clone the Repository

```bash
git clone https://github.com/your-username/sarathi.git
```

```bash
cd sarathi
```

### 🛠 2. Install Requirements

```bash
pip install -r requirements.txt
```

### ▶ 3. Launch the App

```bash
streamlit run driver_state_detection/app.py
```

> ⚠ Make sure your webcam is connected and accessible.

---

## 🧪 Technical Workflow

SARATHI performs **attention analysis** using:
- 👁 **Eye landmarks** to detect prolonged closure or gaze shifts
- 🧠 **Head pose** (pitch, yaw, roll) to monitor orientation
- 🧍 **Posture deviation** using shoulder & spine angles

🧠 **If any threshold is crossed**:
- Screenshot is captured
- Audio alarm is triggered
- Visual status bar updates on Streamlit UI

---

## 🧗‍♂ Challenges Faced

- ⚠ Real-time processing without GPU
- 🌙 Handling low-light conditions
- ❗ Ensuring accuracy while minimizing false alarms
- 📐 Angle-sensitive posture detection
- 🧠 Calibrating thresholds across multiple face orientations

---

## 🌱 Future Enhancements

- 📷 **Dual camera setup** (side + front view) using IoT (e.g., Raspberry Pi)
- 🧘 **User posture calibration** on startup for custom thresholds
- ✋ **Gesture-based alert dismissal** using hand detection
- 🚨 **SOS Trigger**: Auto-message if alert persists beyond limit
- 🌒 **Night mode** via IR camera support

---

## 🏁 Developed At

**Hack4Bengal 4.0 – June 2025**  
Team Name: **Dot Slash**  
Team Lead: **Dipan Mazumder**  
Team Member: **Subhas Pramanik**

---

## 🕉 Why the Name "SARATHI"?

In ancient Indian tradition, **SARATHI** means “charioteer” — the one who safely drives the warrior through chaos.  
In our context, SARATHI acts as a **modern digital guide**, **monitoring attentiveness**, **preventing fatigue**-induced danger, and **saving lives** through intelligence.

---

## 📬 Contact

| 📧 Email | 💻 GitHub | 🔗 LinkedIn |
|----------|-----------|--------------|
| [Dipan Mazumder](mailto:dipanmazumder313@gmail.com)<br>[Subhas Pramanik ](mailto:subhaspramanik38@gmail.com) | [Dipan Mazumder](https://github.com/dipan313)<br>[Subhas Pramanik](https://github.com/subhas-pramanik-09) | [Dipan Mazumder](https://www.linkedin.com/in/dipan-mazumder-953453279/)<br>[Subhas Pramanik](https://www.linkedin.com/in/subhas-pramanik) |

---

> Built to protect. Designed to alert. SARATHI is not just code — it's a life-saving companion.
