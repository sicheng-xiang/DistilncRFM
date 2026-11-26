
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = '/root/autodl-tmp/structRFM-master2/'
def project_path(value):
    text = str(value).replace('\\', '/')
    if text.startswith(OLD_ROOT):
        return str(ROOT / text[len(OLD_ROOT):])
    p = Path(value).expanduser()
    return str(p.resolve() if p.is_absolute() else (ROOT / p).resolve())
