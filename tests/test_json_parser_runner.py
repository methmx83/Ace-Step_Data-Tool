from scripts.helpers.json_parser import JSONParser
import json

cases = [
    ('{"rap_style": ["trap"]}', 'rap_style'),
    ('{"Rap Style": ["lyrical"]}', 'rap_style'),
    ('{"rap-style": ["mumble rap"]}', 'rap_style'),
    ('"{\"Rap Style\": [\"oldschool\"]}"', 'rap_style'),
    ('Some text mentioning trap and mumble rap', 'rap_style'),
]

for txt, cat in cases:
    out = JSONParser.extract_json(txt, cat)
    print(txt, '=>', json.dumps(out, ensure_ascii=False))
