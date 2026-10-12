"""Download documented USGS RDB sources; snapshots may be revised by USGS."""
from pathlib import Path
import json,hashlib,urllib.request
root=Path(__file__).resolve().parent
out=root/'data/raw';out.mkdir(parents=True,exist_ok=True)
for item in json.loads((root/'sources.json').read_text()):
    p=out/item['file']
    if not p.exists():
        with urllib.request.urlopen(item['url'],timeout=90) as response:
            content=response.read()
        if b'agency_cd\tsite_no\tdatetime' not in content:
            raise ValueError('Response is not a USGS RDB data file')
        p.write_bytes(content)
    current=hashlib.sha256(p.read_bytes()).hexdigest()
    print(p.name, 'exact snapshot' if current==item['sha256'] else 'different retrieval bytes: headers/data may have changed')
