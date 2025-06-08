#!/usr/bin/env python3
"""
Download every .mat file from a Box folder, storing each one at DEST_DIR.
For any local file the script checks:

1. Byte count equals the value reported by Box
2. SHA-1 hash matches when Box provides it
3. `scipy.io.whosmat` can read the header

Only files that fail one of those tests, or are missing, are downloaded.
"""

import os
import hashlib
import requests
from scipy.io import whosmat
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
ACCESS_TOKEN = "deuy2E4kmrboh4E7hZhGHKxqVOBjuc22"
FOLDER_ID    = "322150952623"
DEST_DIR     = "/project2/cdonnat/sleepstages/data/overlapping_data"
MAX_THREADS  = 10
CHUNK_SIZE   = 1 << 13
# ----------------------------------------


def sha1_file(path, buf=1 << 16):
    """Return hex SHA-1 for a local file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def list_folder(folder_id, prefix=""):
    """
    Recursively walk a Box folder and return a dict

        {relative_path: {"id": box_file_id, "size": bytes, "sha1": hash_or_None}}
    """
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    url     = f"https://api.box.com/2.0/folders/{folder_id}/items"
    params = {
        "limit": 1000,
        "offset": 0,
        "fields": "id,name,type,size,sha1"
    }
    mapping = {}

    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for item in data["entries"]:
            name = item["name"]
            path = os.path.join(prefix, name)
            if item["type"] == "file":
                mapping[path] = {
                    "id":   item["id"],
                    "size": item["size"],
                    "sha1": item.get("sha1"),  # may be None on some Box plans
                }
            elif item["type"] == "folder":
                mapping.update(list_folder(item["id"], path))

        if len(data["entries"]) < params["limit"]:
            break
        params["offset"] += len(data["entries"])

    return mapping


def download_file(session, meta, rel_path):
    """
    Fetch rel_path if it is missing or corrupted, otherwise skip.
    Corruption is defined as size mismatch, hash mismatch, or header unreadable.
    """
    fid      = meta["id"]
    exp_size = meta["size"]
    exp_sha1 = meta["sha1"]
    dest     = os.path.join(DEST_DIR, rel_path)

    # quick check for an existing local file
    if os.path.exists(dest):
        size_ok  = os.path.getsize(dest) == exp_size
        hash_ok  = exp_sha1 is None or sha1_file(dest) == exp_sha1
        header_ok = False
        if size_ok and hash_ok:
            try:
                whosmat(dest)
                header_ok = True
            except Exception:
                pass

        if size_ok and hash_ok and header_ok:
            return  # file is healthy

        print(f"{rel_path}: local copy failed validation, re downloading")
        try:
            os.remove(dest)
        except FileNotFoundError:
            pass

    # perform the download
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        url = f"https://api.box.com/2.0/files/{fid}/content"
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)

        # verify the new file
        if os.path.getsize(dest) != exp_size:
            raise ValueError("size mismatch after download")
        if exp_sha1 and sha1_file(dest) != exp_sha1:
            raise ValueError("hash mismatch after download")
        whosmat(dest)  # final structure check
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"Download failed for {rel_path}: {e}")


def main():
    print("Indexing Box folder …")
    file_map = list_folder(FOLDER_ID)
    total    = len(file_map)
    print(f"Total files reported by Box: {total}")

    print("Validating local files and downloading any that are missing or bad …")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {ACCESS_TOKEN}"})

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as pool:
        futures = [
            pool.submit(download_file, session, meta, rpath)
            for rpath, meta in file_map.items()
        ]
        for _ in tqdm(as_completed(futures), total=total, desc="Progress"):
            pass

    print("All done.")


if __name__ == "__main__":
    main()
