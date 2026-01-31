from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
from PIL import Image
import io
import cv2
import pytesseract
from typing import Dict, Any
import asyncio

app = FastAPI(title="ID + Face Verification API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← tighten this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recommended: limit upload size (~8MB should be more than enough for IDs/selfies)
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MiB


async def read_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    """Read image bytes and PIL Image safely with size limit"""
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file too large (max 8MB)"
        )

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()  # Validate it's really an image
        img = Image.open(io.BytesIO(contents))  # reopen after verify
        return contents, img
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )


def calculate_entropy_cv2(img_array: np.ndarray) -> float:
    """Calculate entropy from numpy array (grayscale)"""
    if img_array.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist + 1e-10))


@app.post("/validate-id")
async def validate_id(idImage: UploadFile = File(...)):
    if idImage.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "Only JPEG and PNG images are allowed")

    try:
        contents, pil_img = await read_image(idImage)
        img_array = np.array(pil_img.convert("RGB"))

        w, h = pil_img.size
        if min(w, h) < 500:
            return {"valid": False, "message": "Image resolution too low (<500px)"}

        aspect = max(w, h) / min(w, h)
        if not (1.05 <= aspect <= 2.4):
            return {"valid": False, "message": "Unusual aspect ratio for an ID document"}

        # ── Face detection ─────────────────────────────────────────────
        try:
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend="retinaface",
                enforce_detection=True
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

        # Face size check
        face = faces[0]
        fa = face["facial_area"]
        face_ratio = (fa["w"] * fa["h"]) / (w * h)
        if face_ratio > 0.40:  # slightly more relaxed in 2025
            return {"valid": False, "message": "Face too large — please show the FULL ID card"}

        # Entropy (quality/detail check)
        entropy = calculate_entropy_cv2(img_array)
        if entropy < 4.6:
            return {"valid": False, "message": "Image too flat/low-detail (possible screenshot or low quality)"}

        # OCR text presence
        try:
            ocr_text = pytesseract.image_to_string(pil_img.convert("L")).lower().strip()
            keywords = {'name', 'birth', 'date', 'id', 'no.', 'number', 'philippine', 'sss', 'umid',
                        'license', 'passport', 'card', 'philid', 'address'}
            has_text = len(ocr_text) > 25 or any(kw in ocr_text for kw in keywords)
            if not has_text:
                return {"valid": False, "message": "No readable text detected on ID"}
        except:
            pass  # OCR failure is not fatal

        # Emotion check (anti-selfie)
        try:
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if analysis:
                dom = analysis[0]['dominant_emotion']
                score = analysis[0]['emotion'][dom]
                if dom in {'happy', 'surprise'} and score > 78:
                    return {"valid": False, "message": "Strong facial expression detected (looks like selfie)"}
        except:
            pass

        return {"valid": True, "message": "ID photo looks valid ✓"}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"valid": False, "message": "Internal server error during validation"}
        )


@app.post("/verify/face")
async def verify_face(
    idImage: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    if idImage.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "ID image must be JPEG/PNG")
    if selfie.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, "Selfie must be JPEG/PNG")

    try:
        id_bytes, id_pil = await read_image(idImage)
        selfie_bytes, selfie_pil = await read_image(selfie)

        id_array = np.array(id_pil.convert("RGB"))
        selfie_array = np.array(selfie_pil.convert("RGB"))

        # Quick selfie size sanity check
        try:
            faces = DeepFace.extract_faces(selfie_array, detector_backend="retinaface")
            if len(faces) == 1:
                fa = faces[0]["facial_area"]
                ratio = (fa["w"] * fa["h"]) / (selfie_pil.width * selfie_pil.height)
                if ratio > 0.68:
                    return {"success": False, "message": "Selfie taken too close (face fills almost entire frame)"}
        except:
            pass

        # Core face verification
        result = DeepFace.verify(
            img1_path=id_array,
            img2_path=selfie_array,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            silent=True
        )

        return {
            "success": bool(result["verified"]),
            "message": "Faces match ✓" if result["verified"] else "Faces do not match ✗",
            "distance": round(float(result.get("distance", 0)), 4),
            "threshold": round(float(result.get("threshold", 0)), 4)
        }

    except ValueError as e:
        msg = str(e).lower()
        if "could not be detected" in msg:
            return {"success": False, "message": "No clear face detected in one or both images"}
        if "multiple faces" in msg:
            return {"success": False, "message": "Multiple faces detected in one of the images"}
        return {"success": False, "message": "Face verification failed"}
    except Exception:
        return {"success": False, "message": "Internal error during face verification"}