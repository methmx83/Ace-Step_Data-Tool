import json
from scripts.helpers.json_parser import JSONParser

cases = [
    '{"Rap Style": ["lyrical rap"]}',
    '{\\"Rap Style\\": [\\"lyrical rap\\"]}',
    '"{\\"Rap Style\\": [\\"oldschool\\"]}"',
    '{\\\\"Rap Style\\\\": [\\\\"mumble rap\\\\"]}',
]

out_lines = []
for s in cases:
    out_lines.append('INPUT_REPR: ' + repr(s))
    try:
        jl = json.loads(s)
        out_lines.append('json.loads OK -> type:' + str(type(jl)) + ' repr:' + repr(jl))
    except Exception as e:
        out_lines.append('json.loads FAILED: ' + str(e))
    try:
        tj = JSONParser._try_parse_json(s)
        out_lines.append('_try_parse_json -> ' + repr(tj))
    except Exception as e:
        out_lines.append('_try_parse_json EXCEPTION: ' + str(e))
    # try manual unescape
    s2 = s
    s2 = s2.replace('\\"','"')
    out_lines.append('after replace \\\" -> "  repr:' + repr(s2))
    try:
        jl2 = json.loads(s2)
        out_lines.append('json.loads(after replace) OK -> ' + repr(jl2))
    except Exception as e:
        out_lines.append('json.loads(after replace) FAILED: ' + str(e))
    out_lines.append('---')

print('\n'.join(out_lines))
