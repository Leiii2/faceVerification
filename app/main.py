from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2

# ─────────────────────────────────────────────────────────────
# OPTIONAL OCR (does NOT crash if unavailable)
# ─────────────────────────────────────────────────────────────
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="ID + Face Verification API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Health check (REQUIRED for Render)
# ─────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MB


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
async def read_image(file: UploadFile):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "Image file too large (max 8MB)")

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents))
        return img
    except Exception:
        raise HTTPException(400, "Invalid or corrupted image file")


def calculate_entropy_cv2(img_array: np.ndarray) -> float:
    if img_array.size == 0:
        return 0.0

    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


# ─────────────────────────────────────────────────────────────
# ID VALIDATION
# ─────────────────────────────────────────────────────────────
@app.post("/validate-id")
async def validate_id(idImage: UploadFile = File(...)):
    from deepface import DeepFace  # lazy import (IMPORTANT)

    if idImage.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "Only JPEG and PNG images are allowed")

    try:
        pil_img = await read_image(idImage)
        img_array = np.array(pil_img.convert("RGB"))

        w, h = pil_img.size
        if min(w, h) < 500:
            return {"valid": False, "message": "Image resolution too low (<500px)"}

        aspect = max(w, h) / min(w, h)
        if not (1.05 <= aspect <= 2.4):
            return {"valid": False, "message": "Unusual aspect ratio for an ID document"}

        # Face detection
        try:
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend="retinaface",
                enforce_detection=True,
            )
            if len(faces) != 1:
                return {"valid": False, "message": f"Expected exactly 1 face, found {len(faces)}"}
        except ValueError as e:
            msg = str(e).lower()
            if "could not be detected" in msg:
                return {"valid": False, "message": "No clear face detected"}
            if "multiple faces" in msg:
                return {"valid": False, "message": "Multiple faces detected"}
            raise

        fa = faces[0]["facial_area"]
        face_ratio = (fa["w"] * fa["h"]) / (w * h)
        if face_ratio > 0.40:
            return {"valid": False, "message": "Face too large — show the FULL ID"}

        entropy = calculate_entropy_cv2(img_array)
        if entropy < 4.6:
            return {"valid": False, "message": "Image too flat / low quality"}

        # OCR (OPTIONAL)
        if OCR_AVAILABLE:
            try:
                ocr_text = pytesseract.image_to_string(
                    pil_img.convert("L")
                ).lower().strip()

                keywords = {
                    "name", "birth", "date", "id", "number", "philippine",
                    "sss", "umid", "license", "passport", "card", "philid", "address"
                }

                has_text = len(ocr_text) > 25 or any(k in ocr_text for k in keywords)
                if not has_text:
                    return {"valid": False, "message": "No readable text detected on ID"}
            except Exception:
                pass

        # Emotion check
        try:
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if analysis:
                dom = analysis[0]["dominant_emotion"]
                score = analysis[0]["emotion"][dom]
                if dom in {"happy", "surprise"} and score > 78:
                    return {"valid": False, "message": "Looks like a selfie photo"}
        except Exception:
            pass

        return {"valid": True, "message": "ID photo looks valid ✓"}

    except HTTPException:
        raise
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"valid": False, "message": "Internal server error"},
        )


# ─────────────────────────────────────────────────────────────
# FACE VERIFICATION
# ─────────────────────────────────────────────────────────────
@app.post("/verify/face")
async def verify_face(
    idImage: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    from deepface import DeepFace  # lazy import (IMPORTANT)

    if idImage.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "ID image must be JPEG/PNG")
    if selfie.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "Selfie must be JPEG/PNG")

    try:
        id_pil = await read_image(idImage)
        selfie_pil = await read_image(selfie)

        id_array = np.array(id_pil.convert("RGB"))
        selfie_array = np.array(selfie_pil.convert("RGB"))

        # Selfie distance sanity check
        try:
            faces = DeepFace.extract_faces(selfie_array, detector_backend="retinaface")
            if len(faces) == 1:
                fa = faces[0]["facial_area"]
                ratio = (fa["w"] * fa["h"]) / (selfie_pil.width * selfie_pil.height)
                if ratio > 0.68:
                    return {"success": False, "message": "Selfie taken too close"}
        except Exception:
            pass

        result = DeepFace.verify(
            img1_path=id_array,
            img2_path=selfie_array,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            silent=True,
        )

        return {
            "success": bool(result["verified"]),
            "message": "Faces match ✓" if result["verified"] else "Faces do not match ✗",
            "distance": round(float(result.get("distance", 0)), 4),
            "threshold": round(float(result.get("threshold", 0)), 4),
        }

    except ValueError as e:
        msg = str(e).lower()
        if "could not be detected" in msg:
            return {"success": False, "message": "No clear face detected"}
        if "multiple faces" in msg:
            return {"success": False, "message": "Multiple faces detected"}
        return {"success": False, "message": "Face verification failed"}
    except Exception:
        return {"success": False, "message": "Internal error during verification"}
