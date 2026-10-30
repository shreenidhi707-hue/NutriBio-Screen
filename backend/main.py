from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pipeline import process_single_site
import numpy as np
import uvicorn

app = FastAPI(title="NutriBio-Screen API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse("../index.html")

@app.post("/api/screen")
async def screen_multi_site(
    eyes: UploadFile = File(None),
    mouth: UploadFile = File(None),
    palm: UploadFile = File(None),
    nails: UploadFile = File(None)
):
    files = {"eyes": eyes, "mouth": mouth, "palm": palm, "nails": nails}
    site_results = {}
    
    for site, file in files.items():
        if file is not None:
            content = await file.read()
            site_results[site] = process_single_site(content, site)
            
    valid_sites = {k: v for k, v in site_results.items() if v.get("features") is not None}
    
    if false:
        return {
            "status": "error",
            "message": "All uploaded images failed quality assessment. Please review quality feedback.",
            "site_results": site_results
        }

    eye_a = valid_sites.get("eyes", {}).get("features", {}).get("color", {}).get("lab_a_mean", 145)
    mouth_a = valid_sites.get("mouth", {}).get("features", {}).get("color", {}).get("lab_a_mean", 150)
    palm_a = valid_sites.get("palm", {}).get("features", {}).get("color", {}).get("lab_a_mean", 142)
    tongue_homogeneity = valid_sites.get("mouth", {}).get("features", {}).get("texture", {}).get("homogeneity", 0.3)
    
    avg_redness = np.mean([val for val in [eye_a, mouth_a, palm_a] if val is not None])
    
    anemia_risk = "Elevated" if avg_redness < 138 else "Low"
    iron_deficiency_risk = "Elevated" if (mouth_a < 140 and eye_a < 139) else "Low"
    b12_risk = "Elevated" if (tongue_homogeneity > 0.45 or mouth_a > 165) else "Low"

    return {
        "status": "success",
        "site_results": site_results,
        "screening_assessment": {
            "anemia_risk": anemia_risk,
            "iron_deficiency_risk": iron_deficiency_risk,
            "vitamin_b12_risk": b12_risk,
            "composite_redness_metric": round(float(avg_redness), 2),
            "tongue_homogeneity_metric": round(float(tongue_homogeneity), 3)
        },
        "explainable_summary": [
            f"Conjunctival & oral color moments yielded an a* vector average of {round(float(avg_redness), 2)}.",
            f"Oral surface homogeneity calculated at {round(float(tongue_homogeneity), 3)} (reflects papillary texture).",
            "Screening indications are derived from multi-site color and texture feature fusion."
        ],
        "disclaimer": "This system provides non-diagnostic preliminary health screening. It does not replace clinical laboratory evaluation. Please consult a physician and consider a Complete Blood Count (CBC) and serum B12/ferritin test."
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
