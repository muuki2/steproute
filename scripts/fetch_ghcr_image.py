"""Fetch a public GHCR image (linux/arm64) anonymously into a `docker load` archive.

Why: on the maintainer's Mac (13 Nov 2026) `docker pull ghcr.io/epoch-research/...` failed
with "403 Forbidden" on the token request, because Docker Desktop presents the credential it
has stored for ghcr.io and these public repositories refuse it; an anonymous token works.
This fetches the image without any credential and leaves the stored login alone.

Standard library only. Layers download in parallel into BLOBDIR, resume after a dropped
connection (HTTP Range) and are each checked against their sha256 digest; the image config
must say arm64. The archive holds both the OCI layout and Docker's manifest.json.

    uv run python scripts/fetch_ghcr_image.py IMAGE OUT.tar BLOBDIR
    docker load -i OUT.tar && rm -r OUT.tar BLOBDIR
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import sys
import tarfile
import time
import urllib.parse

ACCEPT = ",".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def request(url: str, headers: dict[str, str]) -> http.client.HTTPResponse:
    for _ in range(5):
        u = urllib.parse.urlsplit(url)
        conn = http.client.HTTPSConnection(u.netloc, timeout=120)
        path = u.path + (f"?{u.query}" if u.query else "")
        conn.request("GET", path, headers=headers)
        r = conn.getresponse()
        if r.status in (301, 302, 303, 307, 308):
            url = r.getheader("Location") or ""
            r.read()
            # never forward the registry token to the blob store; keep a Range header
            headers = {k: v for k, v in headers.items() if k == "Range"}
            continue
        if r.status not in (200, 206):
            raise OSError(f"GET {u.netloc}{u.path}: HTTP {r.status} {r.read()[:200]!r}")
        return r
    raise OSError("too many redirects")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def download(repo: str, auth: dict[str, str], desc: dict, blobdir: str) -> str:
    d, size = desc["digest"], int(desc["size"])
    path = os.path.join(blobdir, d.split(":", 1)[1])
    for attempt in range(30):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == size:
            if sha256_file(path) == d:
                return path
            os.remove(path)
            have = 0
        try:
            hdrs = dict(auth)
            if have:
                hdrs["Range"] = f"bytes={have}-"
            r = request(f"https://ghcr.io/v2/{repo}/blobs/{d}", hdrs)
            mode = "ab" if (have and r.status == 206) else "wb"
            with open(path, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except (OSError, http.client.HTTPException) as exc:
            print(f"  retry {attempt + 1} for {d[:19]}: {type(exc).__name__}", flush=True)
            time.sleep(min(30, 2 * (attempt + 1)))
    raise SystemExit(f"could not download {d}")


def main() -> None:
    image, out, blobdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(blobdir, exist_ok=True)
    ref, _, tag = image.rpartition(":")
    host, _, repo = ref.partition("/")
    assert host == "ghcr.io", image
    tok = json.loads(
        request(f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io", {}).read()
    )["token"]
    auth = {"Authorization": f"Bearer {tok}"}

    def get_manifest(r: str) -> tuple[dict, bytes]:
        resp = request(f"https://ghcr.io/v2/{repo}/manifests/{r}", {**auth, "Accept": ACCEPT})
        raw = resp.read()
        return json.loads(raw), raw

    man, raw = get_manifest(tag)
    if "manifests" in man:  # an index: pick linux/arm64
        pick = [
            m
            for m in man["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m.get("platform", {}).get("architecture") == "arm64"
        ]
        if not pick:
            raise SystemExit("no linux/arm64 manifest in the index")
        digest = pick[0]["digest"]
        man, raw = get_manifest(digest)
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == digest
    cfg_desc, layers = man["config"], man["layers"]
    cfg_path = download(repo, auth, cfg_desc, blobdir)
    arch = json.load(open(cfg_path)).get("architecture")
    if arch != "arm64":
        raise SystemExit(f"image config says architecture {arch!r}, not arm64")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda layer: download(repo, auth, layer, blobdir), layers))

    def name(desc: dict) -> str:
        return "blobs/sha256/" + desc["digest"].split(":", 1)[1]

    def add_bytes(tar: tarfile.TarFile, n: str, data: bytes) -> None:
        info = tarfile.TarInfo(n)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(out, "w") as tar:
        tar.add(cfg_path, arcname=name(cfg_desc))
        for desc, p in zip(layers, paths, strict=True):
            tar.add(p, arcname=name(desc))
        mdig = hashlib.sha256(raw).hexdigest()
        add_bytes(tar, f"blobs/sha256/{mdig}", raw)
        mtype = man.get("mediaType", "application/vnd.oci.image.manifest.v1+json")
        index = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": mtype,
                    "digest": f"sha256:{mdig}",
                    "size": len(raw),
                    "annotations": {
                        "io.containerd.image.name": image,
                        "org.opencontainers.image.ref.name": tag,
                    },
                    "platform": {"architecture": "arm64", "os": "linux"},
                }
            ],
        }
        add_bytes(tar, "oci-layout", json.dumps({"imageLayoutVersion": "1.0.0"}).encode())
        add_bytes(tar, "index.json", json.dumps(index).encode())
        dm = [{"Config": name(cfg_desc), "RepoTags": [image], "Layers": [name(x) for x in layers]}]
        add_bytes(tar, "manifest.json", json.dumps(dm).encode())
    total = sum(int(x["size"]) for x in layers)
    print(f"{image}: {len(layers)} layers, {total / 1e9:.2f} GB compressed -> {out}", flush=True)


if __name__ == "__main__":
    main()
