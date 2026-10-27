Real or fake? Live demo for the Applied Vision Transformer Report (Ali Aljawi, 1641487, MAAI901)

This folder holds model weights and dataset faces: keep it private and off GitHub.
Faces: Celeb-DF v2 (Li et al., 2020) and DF40 (Yan et al., 2024; CC BY-NC 4.0); shown for this assessment only.

MAIN: Google Colab
1. Upload this whole folder to Google Drive as  My Drive/ekyc_demo
2. In Drive, right-click demo_gradio.ipynb > Open with > Google Colaboratory
3. Runtime > Run all (a CPU runtime is enough; about 1-2 minutes the first time)
4. Use the app under the last cell, or open its public link in a new tab for full screen

BACKUP: the laptop, no Colab or Wi-Fi needed once installed (set this up before class)
1. Open a terminal in this folder and run:
     python -m venv .venv
     .venv\Scripts\activate
     pip install torch transformers==5.20.2 timm==1.0.30 safetensors==0.8.0 gradio
2. Start the app:
     python demo_app.py
   It opens http://127.0.0.1:7860 in the browser.

LAST RESORT: the 20-second screen recording in the slides (record it with Win+Shift+R in the Snipping Tool).

In class (about 30 seconds): click A (both models say GENUINE, but it is a deepfake), tick "Show the answer",
then click E (both say FAKE) and switch the image quality to JPEG 60 (both now say GENUINE).
