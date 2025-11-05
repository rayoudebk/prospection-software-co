import requests
import json

API_KEY = "0a9d1fd5a40a4053c6501ff07f9371b38cf5078dfb5c12f7"

# List of companies to search
companies = [
    {"country_code": "UK", "search_term": "07895806"},  # Company number
    {"country_code": "BE", "search_term": "0890.021.619"}  # Company number
    {"country_code": "FR", "search_term": "4TPM"}
    {"country_code": "DE", "search_term": "QPLIX GmbH"}
    {"country_code": "FR", "search_term": "JUMP TECHNOLOGY"}
]

headers = {
    "api-key": API_KEY
}

results = []

for company in companies:
    print(f"\n🔍 Searching for {company['country_code']} company: {company['search_term']}")
    
    try:
        response = requests.get(
            "https://api.pappers.in/v1/search",
            params={
                "q": company['search_term'],  # Search term (can be company number or name)
                "country_code": company['country_code'],
                "api_token": API_KEY  # Note: might be api_token instead of header
            },
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            if 'results' in data and len(data['results']) > 0:
                company_data = data['results'][0]
                
                result = {
                    "country": company['country_code'],
                    "search_term": company['search_term'],
                    "name": company_data.get('name'),
                    "company_number": company_data.get('company_number'),
                    "activities": company_data.get('activities', []),
                    "local_activities": company_data.get('local_activities', [])
                }
                
                results.append(result)
                
                # Display the activities
                print(f"✅ Found: {result['name']} (#{result.get('company_number')})")
                
                if result['activities']:
                    print("\n  📋 NACE Activities:")
                    for activity in result['activities']:
                        print(f"    • [{activity.get('code')}] {activity.get('name')}")
                else:
                    print("  No NACE activities found")
                
                if result['local_activities']:
                    print("\n  📋 Local Activities:")
                    for activity in result['local_activities']:
                        code = activity.get('code')
                        name = activity.get('name')
                        classification = activity.get('classification', '')
                        print(f"    • [{code}] {name}")
                        if classification:
                            print(f"      Classification: {classification}")
                else:
                    print("  No local activities found")
            else:
                print(f"  ⚠️  No results found")
                
        else:
            print(f"  ❌ Error: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Network error: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

# Save all results to JSON
if results:
    output_file = "international_companies_activities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to '{output_file}'")
else:
    print("\n⚠️  No results to save")