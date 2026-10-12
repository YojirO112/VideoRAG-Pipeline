from urllib3 import response
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream" : False
        })

    response = r.json()
    print(response)
    return response

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# find similarities of question embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape) 
similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_ind = similarities.argsort()[::-1][0:top_results]
# print(max_ind)
new_df = df.loc[max_ind]
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching recursion part of the DSA playlist. Here are video 
subtitle chunks containing video title, video number, start time in seconds, end time in 
seconds, the text in that :

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
-----------------------------------------
{incoming_query}
User asked this question related to the video chunks you have to answer wehre and how much
content is taught in which video (in which video and at what timestamp) asn guide the user to go to that
particular video. If user asks unrelated question, tell h im that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open ("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
    # print(index, item["title"], item["number"], item["text"], item["start"], item["end"])