import json

def test_empty_json_schema_evaluated_as_schema():
    json_schema = {}
    
    # Valutazione corretta: un dizionario vuoto {} e' un oggetto valido (is not None)
    res = json.dumps(json_schema) if json_schema is not None else None
    
    assert res == "{}", f"Expected '{{}}', got {res}"
