# adpr-llama – ADP-ribosylation Site Prediction Web App

This repository contains a simple Gradio web application for interacting with the **adpr-llama** protein language model.  The model predicts potential ADP-ribosylation (ADPr) sites in a protein sequence.

![header](https://raw.githubusercontent.com/jbenbudd/adpr-llama/main/assets/header.png)

---

## ✨ Features

* **Free-form sequence input** – Paste any amino-acid sequence; the app automatically splits it into 21-residue chunks (with right-padding) required by the model.
* **Site aggregation** – Per-chunk predictions are remapped back to your original sequence and presented as `Sites=<A22,F35,…>`.
* **Inline visualisation** – Predicted sites are highlighted **in red** along your sequence for quick inspection.
* **3-D viewer** – An interactive NGL scene shows each residue (grey) with predicted sites coloured **red** so you can rotate/zoom and inspect their relative positions.
* **Easy deployment** – Designed for [Hugging Face Spaces](https://huggingface.co/spaces) – just push the repo and you’re live.

---

## 🚀 Quick start (local)

```bash
# Create an isolated env (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Launch the Gradio demo
python app.py
```

The app will open at `http://127.0.0.1:7860/`.

---

## 🛠  Repository layout

```
├── app.py          # Gradio UI & inference logic
├── requirements.txt
└── README.md
```

If you add additional assets (icons, docs, etc.) place them in appropriate folders.

---

## 🛰  Deploying to Hugging Face Spaces

1.  Create a new **Space** (type _Gradio_) under your organisation or profile.
2.  Push this repository to the Space:

    ```bash
    git remote add space https://huggingface.co/spaces/jbenbudd/adpr-llama
    git push space main
    ```
3.  The Space will build automatically (using `requirements.txt`) and start the app.

### CI/CD with GitHub Actions (optional)

You can automate the push using a tiny workflow (see below).  Store a **HF_TOKEN** with write access to the Space in your repository secrets.

```yaml
name: Deploy to HF Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: huggingface/hub-action@v1
        with:
          repo: jbenbudd/adpr-llama
          token: ${{ secrets.HF_TOKEN }}
```

---

## 🤝 Contributing

Pull requests are welcome!  Please open an issue to discuss your ideas or improvements.

---

## 📜 License

MIT – see [LICENSE](LICENSE) for details. 