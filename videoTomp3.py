# convert the videos to mp3

import os
import subprocess
files = os.listdir("videos")
# print(files)

for file in files:
    # print (file)
    tutorial_number = file.split(". ")[0]
    file_name = file.split(". ")[1]
    print(tutorial_number , file_name)

    subprocess.run(["ffmpeg", "-i", f"videos/{file}",f"audio_mp3/{tutorial_number}_{file_name}.mp3"])