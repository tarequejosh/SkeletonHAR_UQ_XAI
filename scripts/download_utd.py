import os
import urllib.request
import zipfile
from tqdm import tqdm

URL = "http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip"
TARGET_DIR = os.path.join("data", "utd_mhad", "raw")
ZIP_PATH = os.path.join(TARGET_DIR, "Skeleton.zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading UTD-MHAD Skeleton data from {URL}...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get("Content-Length", 0))
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Skeleton.zip", total=total_size) as t:
                with open(ZIP_PATH, "wb") as f_out:
                    block_size = 1024 * 64
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f_out.write(buffer)
                        t.update(len(buffer))
        print("Download complete.")
    else:
        print(f"Archive already exists at {ZIP_PATH}.")
        
    print(f"Extracting {ZIP_PATH} to {TARGET_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    download_and_extract()
