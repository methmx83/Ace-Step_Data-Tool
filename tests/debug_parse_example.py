from scripts.helpers.json_parser import JSONParser
s_list = [
    '{"Rap Style": ["lyrical rap"]}',
    '"{\\"Rap Style\\": [\\"oldschool\\"]}"',
    '{\\"Rap Style\\": [\\"Mumble Rap\\"]}',
]

for s in s_list:
    print('INPUT:', s)
    res = JSONParser.extract_json(s, 'rap_style')
    print('OUTPUT:', res)
    print('---')
