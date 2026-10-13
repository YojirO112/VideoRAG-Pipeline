# import enum
# import requests
# import os
# import json
# import pandas as pd

# def create_embedding(text_list):
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"]
#     return embedding

# jsons = os.listdir("jsons") #list all the jsons
# my_dicts = []
# chunk_id = 0
# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embedding for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
    
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         # chunk['embedding'] = create_embedding(chunk['text'])
#         chunk['embedding'] = embeddings[i]
#         chunk_id+=1
#         my_dicts.append(chunk)
# # print(my_dicts)
# # a = create_embedding(["Cat sat on the mat", "Harry dances on the mat"])
# # print(a)

# df = pd.DataFrame.from_records(my_dicts)
# print(df)


import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list, batch_size=32):
    """
    Splits text_list into batches to handle any number of chunks safely.
    """
    if not text_list:
        return []
    
    all_embeddings = []
    
    # Process text chunks in batches of `batch_size` (default: 32)
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })
        
        # Raise an exception if HTTP request returned an error status code
        r.raise_for_status()
        res_json = r.json()
        
        # Safeguard against unexpected API payload responses
        if "embeddings" not in res_json:
            raise KeyError(f"API response missing 'embeddings' key. Response: {res_json}")
            
        all_embeddings.extend(res_json["embeddings"])
        
    return all_embeddings

jsons = os.listdir("jsons") # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embedding for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# save this dataframe
joblib.dump(df, "embeddings.joblib")