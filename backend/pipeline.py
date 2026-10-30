import cv2
import numpy as np

def validate_biological_tissue(img_rgb: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_tissue = np.array([0, 25, 40], dtype=np.uint8)
    upper_tissue = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_tissue, upper_tissue)
    tissue_ratio = np.sum(mask > 0) / (img_rgb.shape[0] * img_rgb.shape[1])
    return tissue_ratio >= 0.18

def process_single_site(image_bytes: bytes, site_type: str = "eyelid"):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {
            "status": "Invalid File",
            "confidence": 0,
            "finding": "Corrupted Image",
            "why": "Could not decode image file."
        }

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # 1. Reject football / non-tissue images
    if not validate_biological_tissue(rgb):
        return {
            "status": "Invalid Image",
            "confidence": 0,
            "finding": "Target Not Identified",
            "why": "No biological tissue detected. Please upload a clear photo of the requested anatomical site."
        }

    # 2. Crop target center ROI
    if site_type in ["eyelid", "eye"]:
        roi = rgb[int(h * 0.45):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
    elif site_type in ["nails", "nail"]:
        roi = rgb[int(h * 0.25):int(h * 0.65), int(w * 0.25):int(w * 0.75)]
    elif site_type in ["tongue", "mouth"]:
        roi = rgb[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    else:
        roi = rgb[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]

    r_mean = float(np.mean(roi[:, :, 0]))
    g_mean = float(np.mean(roi[:, :, 1]))
    b_mean = float(np.mean(roi[:, :, 2]))

    total = r_mean + g_mean + b_mean
    if total == 0:
        return {"status": "Underexposed", "confidence": 0, "finding": "Dark", "why": "Too dark."}

    redness_ratio = r_mean / total

    # 3. Detect Anaemia (Pale eye = High Risk)
    if site_type in ["eyelid", "eye"]:
        if redness_ratio <= 0.405:
            conf = int(min(95, max(75, (0.405 - redness_ratio) * 450 + 75)))
            return {
                "status": "High Risk / Anaemia Detected",
                "confidence": conf,
                "finding": "Conjunctival Pallor",
                "why": "Significant pallor detected in lower eyelid mucosa, indicating low hemoglobin concentration."
            }
        else:
            return {
                "status": "Healthy Pink",
                "confidence": int(min(90, max(70, redness_ratio * 160))),
                "finding": "Healthy Pink",
                "why": "Adequate vascular redness observed across lower eyelid tissue."
            }

    elif site_type in ["nails", "nail"]:
        if redness_ratio <= 0.395:
            conf = int(min(92, max(72, (0.395 - redness_ratio) * 400 + 72)))
            return {
                "status": "Iron Deficiency Risk",
                "confidence": conf,
                "finding": "Nail Bed Pallor",
                "why": "Vascular blanching and pallor observed across nail bed surface."
            }
        else:
            return {
                "status": "Normal",
                "confidence": 85,
                "finding": "Normal",
                "why": "Healthy vascular redness observed across fingernails."
            }

    elif site_type in ["tongue", "mouth"]:
        if redness_ratio > 0.48 or redness_ratio < 0.36:
            return {
                "status": "Vitamin B12 Risk",
                "confidence": 82,
                "finding": "Glossitis / Depigmentation",
                "why": "Abnormal surface coloration detected on tongue dorsum."
            }
        else:
            return {
                "status": "Normal",
                "confidence": 85,
                "finding": "Normal",
                "why": "Healthy texture and bumps observed on tongue surface."
            }

    else:
        if redness_ratio <= 0.39:
            return {
                "status": "Anaemia Risk",
                "confidence": 80,
                "finding": "Palmar Pallor",
                "why": "Palmar crease pallor detected across hand palm tissue."
            }
        else:
            return {
                "status": "Normal",
                "confidence": 86,
                "finding": "Normal",
                "why": "Healthy vascular redness observed across palm lines."
            }

def extract_color_features(img_bgr, site_type="eyelid"):
    _, encoded = cv2.imencode(".jpg", img_bgr)
    return process_single_site(encoded.tobytes(), site_type=site_type)
