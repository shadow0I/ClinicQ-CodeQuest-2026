# ClinicQ — Queue Prediction & Crowd Management Engine
**CodeQuest: National Tech Hackathon 2026 (Prototype Submission)**  
**Team CodeBlooded:** Lokesh Sharma, Shubham Garg, Parveen Kumar  
**Track:** Software Development (Organized by DotRepo)

---

## 🚀 Overview
Small clinics run on manual paper registers or basic static token displays. Patients see their token number, but have no visibility into when they will actually be called—spending **1 to 3 hours** trapped inside cramped waiting rooms.

**ClinicQ** replaces captive waiting with:
1. **Dynamic 25th–75th Percentile Wait-Time Forecasting:** Uses an adaptive moving window of actual consultation durations to display a live wait range (e.g., `18–26 min`) and a clear **"Come back at"** timestamp.
2. **Zero-Typing Inline Disruption Alerts:** 1-tap staff controls inject a **🔴 Blinking Red Emergency Indicator** or a **Custom Written Phrase** directly between active queue tokens and immediately shift all downstream estimates.
3. **Interactive Automated Phone Call (IVR) + SMS Fallback:** Covers keypad phones and missed turns. Calls/texts patients to ask *"Are you coming to the clinic?"* and captures DTMF keypad responses:
   - **Option A (Press 2 — Delayed):** Shifts the patient into a later free buffer slot without halting the queue.
   - **Option B (Press 9 / No Reply):** Eliminates the token cleanly and tightens downstream estimates.

---

## 🛠️ Technology Stack
* **Frontend Interface:** Lightweight Mobile-First PWA (`<25KB`), Tailwind CSS, Web Audio & Speech Synthesis IVR Simulator
* **Backend & Real-Time API:** Python **FastAPI** + Uvicorn
* **Statistical Engine & Simulator:** **NumPy** (25th–75th percentile moving window + 30-patient Monte Carlo clinic-day simulator)
* **Cloud Deployment:** GitHub + Render Cloud

---

## 💻 How to Run Locally (For Judges)
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
