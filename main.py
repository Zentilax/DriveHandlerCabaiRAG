"""
Cabai RAG - FastAPI Backend
Deployed on Railway

Pipeline order:
  1. Download Excel files from Drive (in memory)
  2. Transpose + merge Excel into combined_df
  3. Backup existing MongoDB → delete → re-insert from combined_df
  4. Make images public, collect links (in memory)
  5. LLM match image filenames → specimen names
  6. Update MongoDB documents with image fields
"""

import io
import os
import re
import json
import traceback
from datetime import datetime
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
MIN_OVERLAP_RATIO = float(os.getenv("MIN_OVERLAP_RATIO", "0.3"))
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
        raise ValueError(f"Folder '{folder_name}' not found")
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


def download_excel_to_df(service, file_id: str) -> pd.DataFrame:
    """Download an Excel file from Drive into memory and return as DataFrame."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_excel(fh)


# =============================================================================
# EXCEL PROCESSING  (from notebook Cell 5)
# =============================================================================

def normalize_column_name(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", "_", text)


def transpose_excel(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Transpose Excel where:
      - Rows = characteristics (karakter)
      - Columns = specimen names
    Into row-per-specimen records with normalized attribute names.
    """
    # Drop index/number columns
    cols_to_drop = []
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized in ["nomor", "nomer", "no", "number", "unnamed_0"] or str(col).startswith("Unnamed: 0"):
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # First column = karakter, rest = specimen names
    karakter_col = df.columns[0]
    specimen_cols = df.columns[1:]

    records = []
    for specimen in specimen_cols:
        record = {"nama": specimen, "source_file": filename}
        for _, row in df.iterrows():
            karakter_value = row[karakter_col]
            if pd.isna(karakter_value) or str(karakter_value).strip() == "":
                continue
            attr = normalize_column_name(karakter_value)
            value = row[specimen]
            record[attr] = value
        records.append(record)

    return pd.DataFrame(records)


def has_enough_overlap(existing_cols, new_cols, threshold: float) -> bool:
    existing_attrs = set(existing_cols) - {"nama", "source_file"}
    new_attrs = set(new_cols) - {"nama", "source_file"}
    if len(existing_attrs) == 0:
        return True
    overlap = existing_attrs & new_attrs
    return len(overlap) / len(existing_attrs) >= threshold


def process_excel_files(service, job_id: str) -> pd.DataFrame:
    """
    Download all Excel files from Drive in memory, transpose, merge, return combined_df.
    No disk writes — all in memory via BytesIO.
    """
    log(job_id, "📊 Listing Excel files in Drive...")
    files = service.files().list(
        q="mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'",
        fields="files(id, name)"
    ).execute()["files"]

    if not files:
        raise ValueError("No Excel files found in Drive")

    log(job_id, f"📊 Found {len(files)} Excel file(s)")

    combined_df = None
    for f in files:
        log(job_id, f"  ⬇ Downloading: {f['name']}")
        raw_df = download_excel_to_df(service, f["id"])
        row_df = transpose_excel(raw_df, f["name"])

        if combined_df is None:
            combined_df = row_df
        else:
            if not has_enough_overlap(combined_df.columns, row_df.columns, MIN_OVERLAP_RATIO):
                raise ValueError(f"Column overlap <{MIN_OVERLAP_RATIO*100:.0f}% for {f['name']} — cannot merge")
            combined_df = pd.concat([combined_df, row_df], ignore_index=True)

        log(job_id, f"  ✓ {f['name']} → {len(row_df)} specimens")

    log(job_id, f"✅ Combined DataFrame: {len(combined_df)} total specimens")
    return combined_df


# =============================================================================
# MONGODB — EXCEL DATA  (from notebook Cell 6)
# =============================================================================

def replace_mongodb_with_excel(combined_df: pd.DataFrame, collection, job_id: str) -> dict:
    """
    Backup existing collection → delete all → insert from combined_df.
    Backup is kept in memory as a list (no disk write).
    """
    # Step 1: Backup existing data in memory
    log(job_id, "🗂️ Backing up existing MongoDB data (in memory)...")
    existing_docs = list(collection.find({}))
    current_count = len(existing_docs)
    log(job_id, f"  Found {current_count} existing documents")

    # Convert ObjectId to str for the in-memory backup
    backup = []
    for doc in existing_docs:
        doc["_id"] = str(doc["_id"])
        backup.append(doc)

    # Step 2: Delete all existing documents
    log(job_id, "🗑️ Deleting all existing documents...")
    delete_result = collection.delete_many({})
    log(job_id, f"  ✓ Deleted {delete_result.deleted_count} documents")

    # Step 3: Insert new data from combined_df
    log(job_id, f"💾 Inserting {len(combined_df)} new documents...")
    records = combined_df.to_dict("records")

    # Clean NaN values — MongoDB doesn't accept float NaN
    clean_records = []
    for record in records:
        clean = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in record.items()}
        clean_records.append(clean)

    if clean_records:
        insert_result = collection.insert_many(clean_records)
        log(job_id, f"  ✓ Inserted {len(insert_result.inserted_ids)} documents")
    else:
        log(job_id, "  ⚠ No records to insert")

    final_count = collection.count_documents({})
    log(job_id, f"✅ MongoDB now has {final_count} documents (was {current_count})")

    return {
        "previous_count": current_count,
        "new_count": final_count,
        "backup_docs": len(backup),
    }


# =============================================================================
# IMAGE MATCHING  (from drive_handler.py)
# =============================================================================

def fuzzy_match_name(query: str, candidates: list, threshold: float = 0.8) -> tuple:
    best_match = None
    best_score = 0
    query_lower = query.lower()
    for candidate in candidates:
        ratio = SequenceMatcher(None, query_lower, candidate.lower()).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = candidate
    if best_score >= threshold:
        return best_match, best_score
    return None, 0


def get_all_cabai_names(collection) -> tuple:
    cabai_docs = list(collection.find({}, {"_id": 1, "nama": 1}))
    name_to_id = {}
    lowercase_to_original = {}
    names_list = []
    for doc in cabai_docs:
        nama = doc.get("nama")
        if nama:
            name_to_id[nama] = str(doc.get("_id"))
            lowercase_to_original[nama.lower()] = nama
            names_list.append(nama)
    return name_to_id, lowercase_to_original, names_list


def match_image_names_to_cabai(image_names: list, cabai_names: list, client: OpenAI) -> dict:
    if not image_names or not cabai_names:
        return {}

    image_names_formatted = "\n".join([f"{i+1}. {name}" for i, name in enumerate(image_names)])
    cabai_names_formatted = "\n".join([f"- {name}" for name in cabai_names])

    prompt = f"""You are a botanical data matching expert. Match image filenames to cabai (chili pepper) specimen names.

IMAGE FILENAMES (from CSV):
{image_names_formatted}

AVAILABLE CABAI SPECIMEN NAMES (from database):
{cabai_names_formatted}

CRITICAL RULES:
1. ONE IMAGE CAN CONTAIN MULTIPLE SPECIMENS - return ALL matching names as an array
2. Extract ALL specimen names from the image filename
3. Match each extracted name to the closest cabai specimen name
4. Be flexible with spelling variations, capitalization, and typos
5. If NO match can be confidently made, return an empty array

EXAMPLES OF MULTI-SPECIMEN IMAGES:
- "katokkon-gendot-buah.jpg" → matches ["katokkon", "Gendot"] (TWO specimens)
- "rawit-merah-keriting.jpg" → matches ["Rawit", "Cabai Merah Keriting"] (TWO specimens)
- "katokkon-bentuk-daun.jpg" → matches ["katokkon"] (ONE specimen)
- "comparison-katokkon-vs-gendot.jpg" → matches ["katokkon", "Gendot"] (TWO specimens)

SINGLE SPECIMEN EXAMPLES:
- "katokkon-buah.jpg" → matches ["katokkon"]
- "gendot_daun.jpg" → matches ["Gendot"]

Return ONLY valid JSON in this exact format:
{{
  "image_filename_1": {{
    "matched_names": ["exact_name_1", "exact_name_2"],
    "confidence": "high/medium/low",
    "count": 2
  }},
  "image_filename_2": {{
    "matched_names": ["exact_name_1"],
    "confidence": "high",
    "count": 1
  }},
  "image_filename_3": {{
    "matched_names": [],
    "confidence": "none",
    "count": 0
  }}
}}

IMPORTANT:
- matched_names is ALWAYS an array (can be empty, have 1 item, or multiple items)
- Use EXACT specimen names from the available list
- count should match the length of matched_names array"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        return {}


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
    folder_name: Optional[str] = None   # override IMAGE_FOLDER_NAME
    skip_excel: bool = False            # skip Excel → MongoDB step
    skip_images: bool = False           # skip image matching step
    dry_run: bool = False               # don't write to MongoDB


@app.post("/process/run")
def process_run(req: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = "job_1"
    _job_status[job_id] = {"status": "running", "logs": [], "result": None}
    background_tasks.add_task(run_pipeline, job_id, req)
    return {"job_id": job_id, "message": "Pipeline started"}


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

        if not OPENAI_API_KEY or not MONGO_URI:
            raise ValueError("OPENAI_API_KEY and MONGODB_URI must be set")

        mongo_client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_CABAI]

        excel_result = {}
        image_result = {}

        # ── PHASE 1: Excel → MongoDB ──────────────────────────────────────────
        if not req.skip_excel:
            log(job_id, "=" * 50)
            log(job_id, "PHASE 1: Excel Data → MongoDB")
            log(job_id, "=" * 50)

            combined_df = process_excel_files(service, job_id)

            if req.dry_run:
                log(job_id, f"[DRY RUN] Would replace MongoDB with {len(combined_df)} records")
                excel_result = {"dry_run": True, "would_insert": len(combined_df)}
            else:
                excel_result = replace_mongodb_with_excel(combined_df, collection, job_id)
        else:
            log(job_id, "⏭️ Skipping Excel phase")

        # ── PHASE 2: Images → MongoDB ─────────────────────────────────────────
        if not req.skip_images:
            log(job_id, "=" * 50)
            log(job_id, "PHASE 2: Images → MongoDB")
            log(job_id, "=" * 50)

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
                log(job_id, f"[DRY RUN] Would match and update {len(image_rows)} images")
                image_result = {"dry_run": True, "total_images": len(image_rows)}
            else:
                client = OpenAI(api_key=OPENAI_API_KEY)

                name_to_id, lowercase_to_original, cabai_names = get_all_cabai_names(collection)
                log(job_id, f"📊 Found {len(cabai_names)} specimens in DB")

                log(job_id, f"🤖 Matching {len(image_rows)} images with OpenAI...")
                match_map = match_image_names_to_cabai(df_images["name"].tolist(), cabai_names, client)
                log(job_id, "✓ LLM matching complete")

                log(job_id, "Grouping images by specimen...")
                cabai_to_images = {}
                matched_count = 0
                unmatched_images = []
                multi_specimen_images = []
                case_mismatch_fixes = []
                fuzzy_match_fixes = []

                for _, row in df_images.iterrows():
                    image_name = row["name"]
                    public_link = row["public link"]
                    match_info = match_map.get(image_name, {})
                    matched_cabai_names = match_info.get("matched_names", [])

                    if matched_cabai_names:
                        if len(matched_cabai_names) > 1:
                            multi_specimen_images.append({"image": image_name, "specimens": matched_cabai_names})

                        for matched_cabai_name in matched_cabai_names:
                            actual_name = None
                            if matched_cabai_name in name_to_id:
                                actual_name = matched_cabai_name
                            elif matched_cabai_name.lower() in lowercase_to_original:
                                actual_name = lowercase_to_original[matched_cabai_name.lower()]
                                case_mismatch_fixes.append({"llm_returned": matched_cabai_name, "actual_db_name": actual_name, "image": image_name})
                            else:
                                fuzzy_result, similarity = fuzzy_match_name(matched_cabai_name, cabai_names, threshold=0.85)
                                if fuzzy_result:
                                    actual_name = fuzzy_result
                                    fuzzy_match_fixes.append({"llm_returned": matched_cabai_name, "actual_db_name": actual_name, "similarity": similarity, "image": image_name})

                            if actual_name:
                                cabai_to_images.setdefault(actual_name, []).append(public_link)
                                matched_count += 1
                    else:
                        unmatched_images.append(image_name)

                unique_images_matched = len(df_images) - len(unmatched_images)
                log(job_id, f"✓ Total image-specimen matches: {matched_count}")
                log(job_id, f"✓ Unique images matched: {unique_images_matched}/{len(df_images)} ({unique_images_matched/len(df_images)*100:.1f}%)")
                log(job_id, f"✓ Images with multiple specimens: {len(multi_specimen_images)}")
                log(job_id, f"✓ Case-sensitivity fixes: {len(case_mismatch_fixes)}")
                log(job_id, f"✓ Fuzzy matching fixes: {len(fuzzy_match_fixes)}")

                log(job_id, "🧹 Clearing old image fields...")
                sample_doc = collection.find_one({})
                if sample_doc:
                    image_fields = {k: "" for k in sample_doc.keys() if k.startswith("image")}
                    if image_fields:
                        collection.update_many({}, {"$unset": image_fields})
                        log(job_id, f"  ✓ Cleared {len(image_fields)} image fields")

                log(job_id, "💾 Updating cabai documents with images...")
                update_count = 0
                for cabai_name, image_urls in cabai_to_images.items():
                    cabai_id = name_to_id[cabai_name]
                    image_update = {f"image{idx}": url for idx, url in enumerate(image_urls, start=1)}
                    collection.update_one({"_id": ObjectId(cabai_id)}, {"$set": image_update})
                    update_count += 1

                docs_with_images = collection.count_documents({"image1": {"$exists": True}})
                log(job_id, f"✅ Updated {update_count} specimens")
                log(job_id, f"✅ Documents with images: {docs_with_images}")

                if unmatched_images:
                    log(job_id, f"Unmatched images ({len(unmatched_images)}):")
                    for img in unmatched_images[:10]:
                        log(job_id, f"  - {img}")

                image_result = {
                    "total": len(df_images),
                    "unique_matched": unique_images_matched,
                    "total_matches": matched_count,
                    "multi_specimen_count": len(multi_specimen_images),
                    "updated": update_count,
                    "docs_with_images": docs_with_images,
                    "case_fixes": len(case_mismatch_fixes),
                    "fuzzy_fixes": len(fuzzy_match_fixes),
                    "unmatched": len(unmatched_images),
                }
        else:
            log(job_id, "⏭️ Skipping image phase")

        mongo_client.close()

        log(job_id, "=" * 50)
        log(job_id, "✅ PIPELINE COMPLETE")
        log(job_id, "=" * 50)

        _job_status[job_id]["status"] = "done"
        _job_status[job_id]["result"] = {
            "excel": excel_result,
            "images": image_result,
        }

    except Exception as e:
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