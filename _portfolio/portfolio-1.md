---
title: "Data collection and fine-tuning Whisper for low resources languages"
excerpt: 'This project focused on building audio data conversion and fine-tuning Whisper for low resource languages<br/><img src="/images/pf1/Whisper.png" width="500" />'
collection: portfolio
---

___
## **Project Summary**
This internship project focuses on developing a data processing pipeline for fine-tuning the state-of-the-art (SOTA) speech recognition model Whisper on low-resource languages, including various African, Indic, and South American languages. The project involves converting and organizing audio data into the appropriate format required for Whisper, and fine-tuning the model to enhance its performance. The final models are intended to be compatible with publicly available repositories such as Hugging Face and GitHub.

## 1. **Audio data collection and preprocessing**

The initial goal of the project was to collect publicly available audio data. The primary sources included the Mozilla Common Voice corpus and various Hugging Face datasets. Given the low-resource nature of the target languages, additional datasets were sourced from platforms like Mendeley Data, which offers open-access resources for research.

Once collected, most of the audio files—especially those from Common Voice, which are typically in .mp3 format—were converted into .wav format, which is required by Whisper. Alongside the audio conversion, metadata files in .tsv format were generated. Each .tsv file included three key columns: filepath, text, and split (indicating whether the sample belongs to the training, validation, or test set).

Although an earlier version of the dataset was available in .json format, additional processing was performed to convert these into the final .tsv format, ensuring compatibility with the Whisper training pipeline. The following source-code was used for file converstion from .json to .tsv.
```python
import json
import csv
import re

# Read the JSON file
input_file = "/home/kiwoong/Desktop/AIhub/src/stagecoach/data/Chichewa/updated_consolidated_speech_data.json"  # Chichewa
output_file = "data/Chichewa/Chichewa.tsv"  # Output TSV file path

# Open and parse the JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write to a TSV file
with open(output_file, "w", encoding="utf-8", newline='') as tsvfile:
    writer = csv.DictWriter(tsvfile, fieldnames=["filepath", "text", "split"], delimiter='\t')
    writer.writeheader()  # Write header

    # Write rows with updated keys and modified filepath
    for entry in data:
        writer.writerow({
            "filepath": f"data/Chichewa/{entry['FILE_PATH']}",
            "text": re.sub('\n', ' ', entry["TRANSCRIPTION"]),
            "split": entry["Split"]
        })

print(f"Data successfully written to {output_file}")
```

After converting the dataset, the number of hours per each of the dataset was calculated. This includes calculating the number of hours in all training, validation, and test datasets in each language.
```python
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

These outcomes were then utilized into fine-tuning process which was a modified version of the existing `stagecoach` repo provided from the XRI Global team.

## 2. **Finetuning Whisper for low resource languages**

For my second project during my internship at XRI Global, I worked on adapting OpenAI's **Whisper** model for four African and one Indic languages: Yoruba, Chichewa, Hausa, Amharic, and Urdu. My role focused on outperforming existing benchmarks that were available in **Huggingface**.


### Language description

**Yoruba**
- Yoruba is a tonal language spoken by over 20 million people in Nigeria and neighboring countries. It features three distinct tones—high, mid, and low—that can change the meaning of words. The language also has a rich phonological system, including vowel length distinctions and nasalization.
- Total number of hours used in the fine-tuning: 5.13 hrs

**Chichewa**

- Also known as Nyanja, Chichewa is spoken in Malawi and parts of Zambia, Mozambique, and Zimbabwe. It is an agglutinative Bantu language with complex verb morphology, where affixes encode tense, aspect, mood, and subject information.
- Total number of hours used in the fine-tuning: 4.83 hrs

**Hausa**

- Hausa is a Chadic language widely spoken across West Africa. It features vowel harmony, a variety of consonant clusters, and frequent borrowings from Arabic and English. Its phonemic inventory includes implosives and glottalized consonants.
- Total number of hours used in the fine-tuning: 3.92 hrs

**Amharic**

- Amharic is the official working language of Ethiopia and belongs to the Semitic language family. It is written in the Ge’ez script and has a root-and-pattern morphology, with extensive inflectional patterns and a large set of verb forms.
- Total number of hours used in the fine-tuning: 2.30 hrs

**Urdu**

- Urdu is an Indo-Aryan language spoken in Pakistan and India. It uses a modified Perso-Arabic script and shares a significant lexical base with Persian and Arabic. It has a rich poetic tradition and employs elaborate honorifics and register distinctions.
- Total number of hours in the Common Voice: 82 validated hours; However, due to storage issue, 30,000 audio files were selected and used in the project.


## Procedures

From the collected data, I managed to fine-tune Whisper model using both my local-machine and remote GPU that was provided from the XRI Global. Since the amount of resources that can be used in my local machine was restricted, I first started with the `Whisper-base` model and expanded it with `Whisper-large-v3-turbo` for my remote GPU.

While fine-tuning Whisper model, several problems occurred. 

### Problem 1: Audio data too long for Whisper to process
Some of the audio files that were collected from the Common Voice and other sources exceeded the limit of the system. Therefore, I added a `filter_toolong` function in the training configuration to handle excessively long duration of files from the collected audio files.

```python
"filter_toolong": {
                "module": "byLength",
                "name": "ByLength",
                "args": [],
                "kwargs": {
                    "fields": ["source", "target"],
                    "maxLen": 180, #self.filter_len
                }
            }
```
The implementation of the filter_toolong is given in:

```python
class FilterBase:
    def __call__(self, data):
        keep = []
        for x in data:
            if self.use(x):
                keep.append(x)
        return keep


from blocks.filterBlk.filterBase import FilterBase

class ByLength(FilterBase):
    def __init__(self, fields, maxLen):
        self.fields = fields
        self.maxLen = maxLen

    def use(self, sample):
        for f in self.fields:
            if type(sample[f]) == str:
                if len(sample[f]) > self.maxLen:
                    return False
            elif type(sample[f]).__name__ == "Tensor":
                if sample[f].shape[0] > self.maxLen:
                    return False
        return True
```

### Problem 2: No improvement on the performance due to limited data

After filtering the long audio files using the `filter_toolong`, I was able to train Whisper model with different languages. While fine-tuning Whisper on Chichewa, another problem occurred. As the training data was limited, the model did not perform well on the ASR task. Therefore, Tranfer learning approach was used to improve the performance of the model.

Since Chichewa is a Bantu language, I collected subset of Swahili dataset from the Common Voice which has an identical language family. Compared to Chichewa, Swahili is relatively high resource language which has a decent amount of audio data that can be used. After collecting the data, I utilized the preprocessing procedures and merged 30k of the audio files with the existing Chichewa dataset. In here, Swahili data was only used during the training, and Chichewa data was used for training, validation and testing.
The below code demonstrates the merging steps between Chichewa and Swahili dataset. Total number of 46.2 hours of Swahili data were added to the training dataset of the Chichewa.
```python
import pandas as pd

ch = pd.read_csv('Chichewa.tsv', sep = '\t')
sw = pd.read_csv('swahili.tsv', sep = '\t')

sw = sw[:30000]
merged = pd.concat([ch, sw])
merged['split'].value_counts()
'''
split
Train    30267
Test        41
Dev         35
Name: count, dtype: int64
'''


```
