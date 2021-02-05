import os
import json
from app import api
from app import app

app.config["SERVER_NAME"] = "localhost"
app.app_context().__enter__()
import pdb; pdb.set_trace()
with open('docs/source/_static/swagger.json', 'w', encoding='utf-8') as f:
    json.dump(api.__schema__, f, ensure_ascii=False, indent=2)
