PixelForge AI — Image Resolution Enhancer

High-quality image upscaling and enhancement tool designed for product designers, UI/UX creators, and visualization workflows.
Built using a multi-stage image enhancement pipeline with OpenCV, with optional support for Real-ESRGAN deep-learning super-resolution.

PixelForge AI improves low-resolution renders by applying denoising, upscaling, sharpening, contrast enhancement, and quality evaluation metrics.

Features
2× and 4× image upscaling (Lanczos4 / Bicubic)
Edge-preserving noise removal (Bilateral Filter)
Unsharp masking for edge sharpening
CLAHE adaptive contrast enhancement
Texture/detail enhancement pass
PSNR / SSIM quality metrics
Batch processing (multiple images)
ZIP download for batch results
Recent enhancement history
Dark modern UI for designers
Works without GPU
Optional Real-ESRGAN integration
Use Cases
Product render enhancement
UI mockup upscaling
E-commerce image improvement
CAD / 3D visualization cleanup
Screenshot sharpening
Image restoration (basic)
Tech Stack
Python
FastAPI
OpenCV
NumPy
Pillow
scikit-image (SSIM)
Uvicorn
PostgreSQL (optional)
Real-ESRGAN (optional)
Project Structure
PixelForge-AI/
│
├── main.py
├── batch_utils.py
├── requirements.txt
├── README.md
│
├── static/
├── templates/
├── uploads/
├── output/
└── weights/   (optional for Real-ESRGAN)
Run Locally (Development Mode)

No database required.

<<<<<<< HEAD
uvicorn main:app --reload --port 8000
=======
bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

>>>>>>> 1771197 (fix requirements for railway postgres)

Open:

<<<<<<< HEAD
http://localhost:8000
=======

http://localhost:8000

>>>>>>> 1771197 (fix requirements for railway postgres)

Job history will be stored in memory.

Production Mode (PostgreSQL)

Create free database:

https://neon.tech

Add .env

DATABASE_URL=postgresql://user:pass@host/db

Run:

uvicorn main:app --host 0.0.0.0 --port 8000 --env-file .env
API Endpoints
Method	Endpoint	Description
POST	/api/enhance	Single image enhancement
POST	/api/enhance-batch	Multiple images
GET	/api/job/{id}	Job status
GET	/api/batch/{id}	Batch status
GET	/api/preview/{id}	Preview
GET	/api/download/{id}	Download image
GET	/api/batch-download/{id}	Download ZIP
GET	/api/jobs	Recent jobs
GET	/api/batches	Recent batches
Enhancement Pipeline
Input Image
  → Validation
  → Bilateral Filter (Noise removal)
  → Lanczos / Bicubic Upscale
  → Unsharp Mask (Sharpen)
  → CLAHE (Adaptive contrast)
  → Detail Enhance
  → PNG Output

Quality metrics:

PSNR
SSIM
Quality Metrics

PixelForge AI evaluates enhancement quality using:

PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)

These metrics help measure similarity between original and enhanced images.

Real-ESRGAN Upgrade (Optional)

To enable deep-learning super-resolution:

<<<<<<< HEAD
pip install realesrgan torch torchvision

Download weights:
=======
bash
pip install realesrgan torch torchvision

Download weights from:
>>>>>>> 1771197 (fix requirements for railway postgres)

https://github.com/xinntao/Real-ESRGAN/releases

Replace enhance function with Real-ESRGAN inference.

Example:

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

This enables true AI-based super-resolution.

Why This Project

This project was built to provide:

fast enhancement without GPU
deployable on free hosting
real-time preview
batch processing
upgrade path to AI models

It is designed for practical use, not only research.

Author

Tanya Kapoor
BCA — Artificial Intelligence
Python | FastAPI | OpenCV | AI | ML

License

MIT License
