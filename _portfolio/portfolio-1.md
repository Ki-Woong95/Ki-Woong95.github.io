---
title: "Data collection and fine-tuning Whisper for low resources languages"
excerpt: "This project focused on building audio data conversion and fine-tuning Whisper for low resource languages<br/><img src="/images/pf1/Whisper.png" width="500" />
collection: portfolio
---

___
## Project Summary
This internship project focuses on developing a data processing pipeline for fine-tuning the state-of-the-art (SOTA) speech recognition model Whisper on low-resource languages, including various African, Indic, and South American languages. The project involves converting and organizing audio data into the appropriate format required for Whisper, and fine-tuning the model to enhance its performance. The final models are intended to be compatible with publicly available repositories such as Hugging Face and GitHub.

The initial goal of the project was to collect publicly available audio data. The primary sources included the Mozilla Common Voice corpus and various Hugging Face datasets. Given the low-resource nature of the target languages, additional datasets were sourced from platforms like Mendeley Data, which offers open-access resources for research.

Once collected, most of the audio files—especially those from Common Voice, which are typically in .mp3 format—were converted into .wav format, which is required by Whisper. Alongside the audio conversion, metadata files in .tsv format were generated. Each .tsv file included three key columns: filepath, text, and split (indicating whether the sample belongs to the training, validation, or test set).

Although an earlier version of the dataset was available in .json format, additional processing was performed to convert these into the final .tsv format, ensuring compatibility with the Whisper training pipeline. The following source-code was used for file converstion from .json to .tsv.
```
import json
import os
import pandas as pd
json_dir = '/home/kiwoong/Desktop/AIhub/src/stagecoach/data/sw/structured' # Swahili dataset
file_path = []
text = []
split = []


for i in os.listdir(json_dir):
    if i.endswith('.json'):
        with open(base_dir +'/'+ i, 'r') as file:
            try:
                data = json.load(file)  # Load JSON data
                
                # Check if the data is a list
                if isinstance(data, list):
                    for entry in data:
                        file_path.append(entry['audio_filepath'])  # Extract 'audio_filepath'
                        text.append(entry['text'])  # Extract 'text'
                        split.append(i.split('.')[0])  # Extract file name without extension
                elif isinstance(data, dict):
                    file_path.append(data['audio_filepath'])  # Extract 'audio_filepath'
                    text.append(data['text'])  # Extract 'text'
                    split.append(i.split('.')[0])  # Extract file name without extension
                else:
                    print(f"Unexpected data structure in file {i}: {type(data)}")
            except KeyError as e:
                print(f"KeyError in file {i}: {e}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in file {i}: {e}")

dataset = pd.DataFrame({'filepath':file_path,
                        'text': text,
                        'split': split
                        })
dataset.to_csv('sw.tsv', sep='\t', index=False)
```
After converting the dataset, the number of hours per each of the dataset was calculated. This includes calculating the number of hours in all training, validation, and test datasets in each language.
```
import os
import wave

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = num_frames / float(frame_rate)
        return duration

def get_total_wav_duration(directory):
    total_duration = 0
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            count += 1
            file_path = os.path.join(directory, filename)
            total_duration += get_wav_duration(file_path)
            
    # Convert total duration to hours
    total_duration_hours = total_duration / 3600
    avg_dur = total_duration/count
    return total_duration_hours, avg_dur

directory = '/home/kiwoong/Desktop/AIhub/src/stagecoach/data/Yoruba/train' # Calculating number of hours in Yoruba training data 
total_duration_hours, avg_dur = get_total_wav_duration(directory)
print(f"Total duration of all .wav files: {total_duration_hours} hours, avg_dur : {avg_dur}")
```

## Results of the data collection and preprocessing

From the data collection and preprocessing steps, the following directories were created:
- `audio` folder with three subdirectories: `train`, `dev`, `val` containing `.wav` files with a Sampling Rate (SR) 16000
- `language.tsv`: consist of file directories, transcription of the audio files and split

These outcomes were then utilized into fine-tuning process which was a modified version of the existing `stagecoach` repo provided from the XRIGlobal team.

## Finetuning Whisper for low resource languages

