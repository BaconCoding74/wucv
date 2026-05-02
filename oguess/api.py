"""
WuWa Inventory Scanner — FastAPI Web API
Run with: uvicorn api:app --reload --port 8000

Endpoints:
  POST /scan          — scan an uploaded screenshot
  GET  /items         — list all known items in dataset
  POST /items/{hash}  — name/rename an item by hash
  DELETE /items/{hash} — remove an item from dataset
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile, os, shutil
from pathlib import Path

from oguess.detector import (
    scan_screenshot, load_dataset, save_dataset,
    register_item, ItemSlot
)

app = FastAPI(
    title="WuWa Inventory Scanner API",
    description="Computer vision API for Wuthering Waves inventory screenshots",
    version="1.0.0"
)

# Allow all origins for local dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Pydantic models
# ──────────────────────────────────────────────

class NameRequest(BaseModel):
    name: str

class ScanResult(BaseModel):
    index: int
    name: str | None
    quantity: int
    is_new: bool
    hash: str
    bbox: list[int]
    crop_url: str | None


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.post("/scan", response_model=list[ScanResult])
async def scan_image(file: UploadFile = File(...), debug: bool = False):
    """
    Upload a WuWa inventory screenshot (PNG/JPG).
    Returns detected items with name, quantity, and hash.
    """
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        slots: list[ItemSlot] = scan_screenshot(tmp_path, debug=debug)
    finally:
        os.unlink(tmp_path)

    if not slots:
        raise HTTPException(status_code=422, detail="No item slots detected in image.")

    results = []
    for s in slots:
        crop_url = f"/crops/{Path(s.crop_path).name}" if s.crop_path else None
        results.append(ScanResult(
            index=s.index,
            name=s.name,
            quantity=s.quantity,
            is_new=s.is_new,
            hash=s.image_hash,
            bbox=list(s.bbox),
            crop_url=crop_url,
        ))
    return results


@app.get("/items")
async def list_items():
    """List all named items in the dataset."""
    db = load_dataset()
    return [
        {"hash": k, "name": v["name"], "seen": v["seen"]}
        for k, v in db.items()
    ]


@app.post("/items/{item_hash}")
async def name_item(item_hash: str, body: NameRequest):
    """Name or rename an item by its perceptual hash."""
    db = load_dataset()
    db = register_item(item_hash, body.name, db)
    return {"status": "ok", "hash": item_hash, "name": body.name}


@app.delete("/items/{item_hash}")
async def delete_item(item_hash: str):
    """Remove an item entry from the dataset."""
    db = load_dataset()
    if item_hash not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    del db[item_hash]
    save_dataset(db)
    return {"status": "deleted", "hash": item_hash}


@app.get("/crops/{filename}")
async def serve_crop(filename: str):
    """Serve unknown item crop images for display in UI."""
    crop_path = Path(__file__).parent / "item_crops" / filename
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(str(crop_path))


@app.get("/health")
async def health():
    db = load_dataset()
    return {"status": "ok", "known_items": len(db)}