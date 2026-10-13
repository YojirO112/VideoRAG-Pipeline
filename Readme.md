# How to use this RAG AI Teching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder. 
## Step 2 - Convert to mp3
Convert all the video files to mp3 by running videoTomp3.py

## Step 3 - Convert mp3 to JSON
Convert all the mp3 files to json by running mp3ToJsons

## step 4 - Convert json files to vectors
Use the file preprocessJson to convert the json files to dataframe with embeddings and save it as a joblib pickle


## step 5 - Prompt generation and feeding to the LLM
Reat the joblin file and load it into the memory. Then
create a relevant prompt as per the user query and feed it to the LLM