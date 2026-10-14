"""Baja el archive.zip completo de la competencia (~265 GB) a data/archive.zip.

Reanudable: continúa desde el tamaño ya escrito con HTTP Range. Si la URL firmada caduca o la
conexión se cae, pide una URL nueva y sigue. Termina verificando tamaño e índice del zip.
Uso: KAGGLE_API_TOKEN=... python scripts/download_archive.py
"""
import os, sys, time, zipfile
from pathlib import Path
import requests
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

COMP = "rsna-knee-abnormality-detection"
OUT = Path(__file__).resolve().parent.parent / "data" / "archive.zip"
CHUNK = 8 << 20
MAX_FAILS = 20


def signed_url():
    with KaggleClient(api_token=os.environ["KAGGLE_API_TOKEN"]) as k:
        r = ApiDownloadDataFilesRequest(); r.competition_name = COMP
        resp = k.competitions.competition_api_client.download_data_files(r)
        url = resp.url; resp.close()
    return url


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    url = signed_url()
    total = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    print(f"tamaño remoto {total / 1e9:.1f} GB", flush=True)
    fails = 0
    while (done := OUT.stat().st_size if OUT.exists() else 0) < total:
        try:
            with requests.get(url, headers={"Range": f"bytes={done}-"}, stream=True, timeout=120) as r:
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code} (se esperaba 206)")
                t0, got, next_log = time.time(), 0, 2 << 30
                with open(OUT, "ab") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk); got += len(chunk)
                        if got >= next_log:  # una línea cada ~2 GB
                            next_log += 2 << 30
                            print(f"{(done + got) / 1e9:.1f}/{total / 1e9:.1f} GB "
                                  f"{got / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
            fails = 0
        except Exception as e:
            fails += 1
            print(f"corte ({type(e).__name__}: {e}); intento {fails}/{MAX_FAILS}, URL nueva", flush=True)
            if fails >= MAX_FAILS:
                sys.exit("demasiados cortes seguidos")
            time.sleep(min(60, 5 * fails))
            url = signed_url()
    if OUT.stat().st_size != total:
        sys.exit(f"tamaño final {OUT.stat().st_size} != {total}")
    n = len(zipfile.ZipFile(OUT).infolist())
    print(f"listo: {OUT} ({total / 1e9:.1f} GB, {n} miembros)", flush=True)


if __name__ == "__main__":
    main()
