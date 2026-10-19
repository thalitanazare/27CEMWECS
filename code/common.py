"""Repository paths and JSON output."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
