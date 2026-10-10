import whisper
import json

model = whisper.load_model("small")

result = model.transcribe(audio = "audio_mp3/1_What is Recursion_3f.mp4.mp3",
                            language="hi",
                            task = "translate",
                            fp16=True)

# print(result["segments"])
# with open("output.json", "w") as f:
#     json.dump(f, result)
chunks = []
for segment in result["segments"] :
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)