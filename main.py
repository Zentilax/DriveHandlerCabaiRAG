"""
Cabai RAG - FastAPI Backend
Deployed on Railway
"""

import io
import os
import json
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd
from bson.objectid import ObjectId
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI
from pymongo import MongoClient
from pydantic import BaseModel

# =============================================================================
# CONFIG
# =============================================================================

SCOPES = ["https://www.googleapis.com/auth/drive"]
IMAGE_FOLDER_NAME = os.getenv("IMAGE_FOLDER_NAME", "gambar")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MONGO_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("DB_NAME", "")
COLLECTION_CABAI = os.getenv("COLLECTION_NAME", "")
RAILWAY_PUBLIC_URL = os.getenv("RAILWAY_PUBLIC_URL", "http://localhost:8000")
GOOGLE_CLIENT_CONFIG = os.getenv("GOOGLE_CLIENT_CONFIG", "")

# In-memory stores
_token_store: dict = {}
_job_status: dict = {}

# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="Cabai RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GOOGLE AUTH HELPERS
# =============================================================================

def get_client_config() -> dict:
    if not GOOGLE_CLIENT_CONFIG:
        raise HTTPException(500, "GOOGLE_CLIENT_CONFIG env var not set")
    config = json.loads(GOOGLE_CLIENT_CONFIG)
    # Normalize "installed" type to "web" for server-side flow
    if "installed" in config:
        config["web"] = config.pop("installed")
        config["web"]["redirect_uris"] = [f"{RAILWAY_PUBLIC_URL}/auth/callback"]
    return config


def get_drive_service():
    creds_data = _token_store.get("default")
    if not creds_data:
        raise HTTPException(401, "Not authenticated. Please visit /auth/login first.")
    creds = Credentials(**creds_data)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _token_store["default"] = credentials_to_dict(creds)
    return build("drive", "v3", credentials=creds)


def credentials_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }


# =============================================================================
# DRIVE UTILITIES
# =============================================================================

def find_folder_by_name(service, folder_name: str) -> str:
    query = (
        f"name = '{folder_name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    res = service.files().list(q=query, fields="files(id, name)").execute()
    if not res["files"]:
        raise ValueError(f"Folder '{folder_name}' not found in Drive")
    return res["files"][0]["id"]


def list_files_in_folder(service, folder_id: str):
    query = f"'{folder_id}' in parents and trashed = false"
    res = service.files().list(
        q=query, fields="files(id, name, mimeType)"
    ).execute()
    return res["files"]


def make_file_public(service, file_id: str):
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"}
    ).execute()


def get_public_link(service, file_id: str) -> str:
    file = service.files().get(fileId=file_id, fields="webViewLink").execute()
    return file["webViewLink"]


# =============================================================================
# MATCHING LOGIC
# =============================================================================

def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.85):
    best_score = 0
    best_match = None
    for candidate in candidates:
        score = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def match_image_names_to_cabai(image_names: list, cabai_names: list, client: OpenAI) -> dict:
    prompt = f"""You are given a list of image filenames and a list of chili specimen (cabai) names.
Match each image filename to one or more specimen names it likely belongs to.
Return a JSON object where keys are image filenames and values are objects with 'matched_names' (list of matching specimen names, or empty list if none match).
Only use names from the provided specimen list.

Image filenames:
{json.dumps(image_names)}

Specimen names:
{json.dumps(cabai_names)}

Return ONLY valid JSON, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def get_all_cabai_names(collection):
    docs = list(collection.find({}, {"name": 1}))
    name_to_id = {doc["name"]: str(doc["_id"]) for doc in docs if "name" in doc}
    lowercase_to_original = {k.lower(): k for k in name_to_id}
    return name_to_id, lowercase_to_original, list(name_to_id.keys())


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================

@app.get("/auth/login")
def auth_login():
    config = get_client_config()
    redirect_uri = f"{RAILWAY_PUBLIC_URL}/auth/callback"
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return RedirectResponse(auth_url)


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(code: str, state: Optional[str] = None):
    config = get_client_config()
    redirect_uri = f"{RAILWAY_PUBLIC_URL}/auth/callback"
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    creds = flow.credentials
    _token_store["default"] = credentials_to_dict(creds)
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Auth Success</title>
      <style>
        body { background: #0d0d0d; color: #f0ebe3; font-family: monospace;
               display: flex; align-items: center; justify-content: center;
               height: 100vh; margin: 0; flex-direction: column; gap: 1rem; }
        .dot { width: 12px; height: 12px; border-radius: 50%; background: #4caf7d;
               box-shadow: 0 0 16px #4caf7d; }
      </style>
    </head>
    <body>
      <div class="dot"></div>
      <p>Connected to Google Drive.</p>
      <p style="color:#7a7167;font-size:0.8rem">You can close this tab.</p>
      <script>
        if (window.opener) {
          window.opener.postMessage('auth_success', '*');
          setTimeout(() => window.close(), 1500);
        }
      </script>
    </body>
    </html>
    """)


@app.get("/auth/status")
def auth_status():
    creds_data = _token_store.get("default")
    if not creds_data:
        return {"authenticated": False}
    creds = Credentials(**creds_data)
    return {"authenticated": creds.valid or bool(creds.refresh_token)}


@app.post("/auth/logout")
def auth_logout():
    _token_store.clear()
    return {"message": "Logged out"}


# =============================================================================
# DRIVE ENDPOINTS
# =============================================================================

@app.get("/drive/files")
def list_drive_files(query: Optional[str] = None, page_size: int = 20):
    service = get_drive_service()
    response = service.files().list(
        q=query,
        pageSize=page_size,
        fields="files(id,name,mimeType,parents)"
    ).execute()
    return {"files": response.get("files", [])}


@app.get("/drive/folders")
def list_folders():
    service = get_drive_service()
    query = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    response = service.files().list(
        q=query, pageSize=50, fields="files(id,name)"
    ).execute()
    return {"folders": response.get("files", [])}


# =============================================================================
# PROCESSING ENDPOINTS
# =============================================================================

class ProcessRequest(BaseModel):
    folder_name: Optional[str] = None
    dry_run: bool = False


@app.post("/process/images")
def process_images(req: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = "job_1"
    _job_status[job_id] = {"status": "running", "logs": [], "result": None}
    background_tasks.add_task(run_pipeline, job_id, req)
    return {"job_id": job_id, "message": "Processing started"}


@app.get("/process/status/{job_id}")
def get_job_status(job_id: str):
    status = _job_status.get(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


def log(job_id: str, msg: str):
    _job_status[job_id]["logs"].append(msg)
    print(msg)


def run_pipeline(job_id: str, req: ProcessRequest):
    try:
        service = get_drive_service()
        folder_name = req.folder_name or IMAGE_FOLDER_NAME

        log(job_id, f"🔍 Finding folder: {folder_name}")
        folder_id = find_folder_by_name(service, folder_name)
        files = list_files_in_folder(service, folder_id)

        log(job_id, f"📁 Found {len(files)} files in folder")

        image_rows = []
        for f in files:
            if f["mimeType"] in ["image/jpeg", "image/png"]:
                make_file_public(service, f["id"])
                link = get_public_link(service, f["id"])
                image_rows.append({"name": f["name"], "public link": link})
                log(job_id, f"  ✓ {f['name']}")

        log(job_id, f"✅ {len(image_rows)} images made public")

        df_images = pd.DataFrame(image_rows)

        if req.dry_run:
            _job_status[job_id]["status"] = "done"
            _job_status[job_id]["result"] = {
                "total_images": len(image_rows),
                "dry_run": True,
                "images": image_rows,
            }
            return

        if not OPENAI_API_KEY or not MONGO_URI:
            raise ValueError("OPENAI_API_KEY and MONGODB_URI must be set for full processing")

        client = OpenAI(api_key=OPENAI_API_KEY)

        log(job_id, "🔌 Connecting to MongoDB...")
        mongo_client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_CABAI]

        name_to_id, lowercase_to_original, cabai_names = get_all_cabai_names(collection)
        log(job_id, f"📊 Found {len(cabai_names)} specimens in DB")

        image_names = df_images["name"].tolist()
        log(job_id, "🤖 Matching images with OpenAI...")
        match_map = match_image_names_to_cabai(image_names, cabai_names, client)

        cabai_to_images = {}
        unmatched = []
        matched_count = 0

        for _, row in df_images.iterrows():
            img_name = row["name"]
            link = row["public link"]
            match_info = match_map.get(img_name, {})
            matched_names = match_info.get("matched_names", [])

            if not matched_names:
                unmatched.append(img_name)
                continue

            for mname in matched_names:
                actual = None
                if mname in name_to_id:
                    actual = mname
                elif mname.lower() in lowercase_to_original:
                    actual = lowercase_to_original[mname.lower()]
                else:
                    fuzzy_result, _ = fuzzy_match_name(mname, cabai_names, 0.85)
                    if fuzzy_result:
                        actual = fuzzy_result

                if actual:
                    cabai_to_images.setdefault(actual, []).append(link)
                    matched_count += 1

        log(job_id, f"🔗 {matched_count} image-specimen links created")
        log(job_id, "🧹 Clearing old image fields...")

        sample_doc = collection.find_one({})
        if sample_doc:
            image_fields = {k: "" for k in sample_doc if k.startswith("image")}
            if image_fields:
                collection.update_many({}, {"$unset": image_fields})

        log(job_id, "💾 Updating specimens in MongoDB...")
        update_count = 0
        for cname, urls in cabai_to_images.items():
            cid = name_to_id[cname]
            update_fields = {f"image{i+1}": url for i, url in enumerate(urls)}
            collection.update_one({"_id": ObjectId(cid)}, {"$set": update_fields})
            update_count += 1

        mongo_client.close()
        log(job_id, f"✅ Updated {update_count} specimens")

        _job_status[job_id]["status"] = "done"
        _job_status[job_id]["result"] = {
            "total_images": len(image_rows),
            "matched": matched_count,
            "unmatched": len(unmatched),
            "updated_specimens": update_count,
        }

    except Exception as e:
        import traceback
        _job_status[job_id]["status"] = "error"
        _job_status[job_id]["logs"].append(f"❌ ERROR: {e}")
        _job_status[job_id]["logs"].append(traceback.format_exc())


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/")
def root():
    return {"message": "Cabai RAG API is running 🌶️", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
