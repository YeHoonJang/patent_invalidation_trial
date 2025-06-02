import requests
import json
import os
from tqdm import tqdm


# total 176627
n = 176627
quantity = 250 # max: 500
file_name = "doc_identifier.jsonl"
root_dir = os.getcwd()
di_path = os.path.join(root_dir, "data", file_name)

doc_url = "https://developer.uspto.gov/ptab-api/decisions/json"

for i in tqdm(range(0, n, quantity), desc="Get Decisions ... "):

    payload = {
        "facetMap": {
            "decisionTypeCategory": ["decision"]
        },
        "recordTotalQuantity": quantity, # limit max: 500
        "recordStartNumber": i # start index
    }

    response = requests.post(doc_url, json=payload)
    data = response.json()
    results = data.get("results", [])

    ## Save to JSONL
    # for j in range(len(results)):
    #     with open(di_path, 'a', encoding='utf-8') as f:
    #         json_record = json.dumps(results[j], ensure_ascii=False)
    #         f.write(json_record+'\n')
