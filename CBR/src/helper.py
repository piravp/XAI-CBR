import json

def pprint(raw_json):
    """
        Pretty print JSON
    """
    return json.dumps(raw_json, indent=4, sort_keys=True)