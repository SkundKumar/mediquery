import pytest
import json
from query_mediquery import handler  # <-- Fixed the import name!

def test_api_schema_validation():
    """TEST 1: Ensures the function always returns a valid HTTP schema, even on failure."""
    mock_event = {"body": json.dumps({"question": "What is Glaucoma?"})}
    
    # We pass None for context since we are testing locally
    response = handler(mock_event, None)
    
    assert "statusCode" in response, "API missing statusCode"
    assert "body" in response, "API missing body payload"
    assert type(response["statusCode"]) == int, "Status code must be an integer"

def test_headers_present():
    """TEST 2: Ensures frontend React/Streamlit apps receive correct content types."""
    mock_event = {"body": json.dumps({"question": "test"})}
    response = handler(mock_event, None)
    
    # Even if it crashes locally due to missing AWS keys, it must return headers
    assert "headers" in response, "API missing headers entirely"
    assert "Content-Type" in response["headers"], "Missing Content-Type header"
    assert response["headers"]["Content-Type"] == "application/json", "Content-Type must be JSON"

def test_empty_payload_handling():
    """TEST 3: Ensures the API doesn't violently crash if a user sends an empty request."""
    mock_event = {} # Completely empty request
    response = handler(mock_event, None)
    
    assert response["statusCode"] in [200, 400, 500], "API did not return a safe fallback status code"
    assert "body" in response, "API did not return a body on empty payload"