import whisper
import json
import os

model = whisper.load_model("small")

audios = os.listdir("audio_mp3")


for audio in audios:
    print(audio)
    if("_" in audio):
        numbers = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        # print(numbers, title)
        result = model.transcribe(audio = f"audio_mp3/{audio}",
                                language="hi",
                                task = "translate",
                                fp16=True)
        
        chunks = []
        for segment in result["segments"] :
            chunks.append({"number": numbers, "title": title ,"start": segment["start"], "end": segment["end"], "text": segment["text"]})

        chunks_with_metaData = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metaData, f)
    