# Delta Training Video Skill-Labeling Pipeline

Automated end-to-end pipeline that

1. **Extracts a transcript** from every `.mp4` video in `resources/videos/`
2. **Generates an image-based summary** of each video
3. **Classifies 3–5 relevant skills** and proficiency levels for every video
4. Saves the enriched metadata to `resources/final_metadata.json`

The core logic lives in **`pipeline.py`**, powered by OpenAI models, MoviePy, OpenCV, and a small helper for log-probability based confidence scores.

---

## Quick start

### 1. Prerequisites

| Requirement              | Tested version | Notes                         |
| ------------------------ | -------------- | ----------------------------- |
| **Python**               | 3.10 – 3.12    |                               |
| **FFmpeg**               | ≥ 5.1          | MoviePy needs this in `$PATH` |
| **pip** / **virtualenv** | latest         | strongly recommended          |

### 2. Clone and install

```bash
git clone https://github.com/<your-org>/<repo>.git
cd <repo>

# create isolated environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# install Python deps
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python pipeline.py
```
