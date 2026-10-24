import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def check_image_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = bool(lap_var < 60.0)
    
    mean_brightness = float(np.mean(gray))
    is_underexposed = bool(mean_brightness < 40.0)
    is_overexposed = bool(mean_brightness > 225.0)
    
    passed = not (is_blurry or is_underexposed or is_overexposed)
    
    reason = "Optimal Quality"
    if is_blurry:
        reason = "Image is too blurry. Please hold steady."
    elif is_underexposed:
        reason = "Image is too dark. Increase lighting."
    elif is_overexposed:
        reason = "Image has severe flash glare/overexposure."
        
    return {
        "passed": passed,
        "laplacian_variance": round(lap_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "reason": reason
    }

def normalize_illumination_grayworld(img_bgr):
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    
    b = np.clip(b * (gray_mean / (b_mean + 1e-5)), 0, 255)
    g = np.clip(g * (gray_mean / (g_mean + 1e-5)), 0, 255)
    r = np.clip(r * (gray_mean / (r_mean + 1e-5)), 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

def mask_specular_glare(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = cv2.inRange(hsv, np.array([0, 0, 215]), np.array([180, 45, 255]))
    non_glare_mask = cv2.bitwise_not(glare_mask)
    return non_glare_mask

def extract_color_features(img_bgr, valid_mask):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    r_vals = rgb[:, :, 0][valid_mask > 0]
    g_vals = rgb[:, :, 1][valid_mask > 0]
    b_vals = rgb[:, :, 2][valid_mask > 0]
    
    l_vals = lab[:, :, 0][valid_mask > 0]
    a_vals = lab[:, :, 1][valid_mask > 0]
    b_lab_vals = lab[:, :, 2][valid_mask > 0]
    s_vals = hsv[:, :, 1][valid_mask > 0]
    
    if len(r_vals) == 0:
        return {}
        
    total_rgb = r_vals.astype(float) + g_vals.astype(float) + b_vals.astype(float) + 1e-5
    norm_red = np.mean(r_vals / total_rgb)
    
    return {
        "lab_a_mean": round(float(np.mean(a_vals)), 2),
        "lab_l_mean": round(float(np.mean(l_vals)), 2),
        "lab_b_mean": round(float(np.mean(b_lab_vals)), 2),
        "norm_red_index": round(float(norm_red), 3),
        "hsv_saturation_mean": round(float(np.mean(s_vals)), 2)
    }

def extract_texture_features(img_bgr, valid_mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    
    return {
        "contrast": round(float(contrast), 2),
        "homogeneity": round(float(homogeneity), 3),
        "energy": round(float(energy), 3)
    }

def process_single_site(image_bytes, site_name):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image payload"}
        
    q_metrics = check_image_quality(img)
    if not q_metrics["passed"]:
        return {
            "site": site_name,
            "quality": q_metrics,
            "features": None
        }
        
    valid_mask = mask_specular_glare(img)
    normalized_img = normalize_illumination_grayworld(img)
    color_feats = extract_color_features(normalized_img, valid_mask)
    texture_feats = extract_texture_features(normalized_img, valid_mask)
    
    return {
        "site": site_name,
        "quality": q_metrics,
        "features": {
            "color": color_feats,
            "texture": texture_feats
        }
    }