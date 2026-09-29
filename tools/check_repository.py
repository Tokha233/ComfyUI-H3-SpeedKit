"""Check JSON, local documentation links and selected release-scope hazards."""

from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
errors = []
count = 0
files = [p for p in ROOT.rglob("*") if p.is_file()
         and not any(part in {".git", "__pycache__", ".venv", "build", "dist"} for part in p.relative_to(ROOT).parts)]
for file in files:
    relative = str(file.relative_to(ROOT))
    if file.suffix in {".safetensors", ".pt", ".so", ".dll", ".mp4", ".wav"}:
        errors.append(f"Unexpected runtime/private artifact: {relative}")
    if file.suffix == ".json":
        try:
            json.loads(file.read_text())
        except ValueError as exc:
            errors.append(f"Invalid JSON {relative}: {exc}")
    if file.suffix not in {".md", ".html"}:
        continue
    text = file.read_text()
    if file.suffix == ".md":
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        links = re.findall(r"\]\(([^)]+)\)", text)
    else:
        links = re.findall(r'''(?:href|src)=["']([^"']+)["']''', text)
    ids = set(re.findall(r'''\bid=["']([^"']+)["']''', text))
    for target in links:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        if url.path:
            resolved = (file.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"Broken/outside link: {relative} -> {target}")
        elif url.fragment and file.suffix == ".html" and url.fragment not in ids:
            errors.append(f"Broken anchor: {relative} -> {target}")
        count += 1
    if re.search(r"/(?:Users|home)/[^/\s]+/|/var/lib/data|-----BEGIN .*PRIVATE KEY-----", text):
        errors.append(f"Machine-specific path or credential marker: {relative}")
status = json.loads((ROOT / "configs/release-status.json").read_text())
if status["acceleration_node"]:
    for required in ("h3_speedkit/runtime.py", "h3_speedkit/video.py", "h3_speedkit/export.py", "tools/build_kernels.py", "evidence/kitchen036-comparison.json"):
        if not (ROOT / required).is_file():
            errors.append("Missing release implementation/evidence: " + required)
    index = json.loads((ROOT / "evidence/kernel-provenance.json").read_text())
    for record in index["files"]:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append("Kernel differs from provenance index: " + record["path"])
provenance = json.loads((ROOT / "evidence/import-provenance.json").read_text())
for entry in provenance["files"]:
    if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["derived_sha256"]:
        errors.append(f"Imported evidence changed: {entry['path']}")
print(json.dumps({"status": "failed" if errors else "passed", "files": len(files),
                  "local_links_checked": count, "errors": errors}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
