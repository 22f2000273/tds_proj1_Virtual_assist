#!/usr/bin/env python3
"""
Test script for the RAG API
This script will test both local and deployed versions of your API
"""

import requests
import json
import time

def test_api_endpoint(url, test_name):
    """Test a specific API endpoint"""
    print(f"\n=== Testing {test_name} ===")
    print(f"URL: {url}")
    
    # Test data
    test_data = {
        "question": "What is the sample content in the database?",
        "image": None
    }
    
    try:
        print("Sending request...")
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ Success!")
                print(f"Answer: {data.get('answer', 'No answer')}")
                print(f"Links: {len(data.get('links', []))} links found")
                
                # Pretty print the response
                print("\nFull Response:")
                print(json.dumps(data, indent=2))
                
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - server may be down")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    health_url = f"{base_url}/health"
    print(f"\n=== Testing Health Endpoint ===")
    print(f"URL: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ Health check passed!")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print("❌ Invalid JSON response from health endpoint")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")

def main():
    """Main test function"""
    print("RAG API Test Tool")
    print("=" * 40)
    
    # Test endpoints
    endpoints = [
        ("Local Development", "http://localhost:8000/api/"),
        ("Vercel Deployment", "https://tds-proj1-virtual-assist.vercel.app/api/"),
    ]
    
    for name, url in endpoints:
        test_api_endpoint(url, name)
        
        # Test health endpoint for this base URL
        base_url = url.replace("/api/", "")
        test_health_endpoint(base_url)
        
        print("\n" + "-" * 50)
        time.sleep(1)  # Brief pause between tests
    
    print("\n=== Test Summary ===")
    print("If tests are failing:")
    print("1. Check that your API_KEY environment variable is set")
    print("2. Ensure the database exists and has the correct schema")
    print("3. Verify the server is running and accessible")
    print("4. Check the server logs for detailed error messages")

if __name__ == "__main__":
    main()