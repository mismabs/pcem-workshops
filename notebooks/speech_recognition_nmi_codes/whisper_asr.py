"""
Open AI Whisper is an AI model that converts spoken language to text. 
Whisper is trained with multilingual and multitask data collected from the web. 
Whisper is resilient to accents, background noise
"""

# Python script to transcribe/translate input audio file (.wav) to transcript in .srt and .json format
# Github reference/README: https://github.com/openai/whisper/tree/main

import os
import pandas as pd
import torch
import whisper
from whisper.utils import get_writer


# Specify audio directory and audio file (default: current directory './data')
audio_dir = './data' 
audio_file = 'single_speaker' # no file extension


# Specify model and language
# Check this page for the list of supported model and language: https://github.com/openai/whisper#available-models-and-languages
model = whisper.load_model("medium")
language = 'Tagalog' # this also supports Taglish

# Specify task
# 	a. transcribe - write speech transcript 
#.  b. translate - write speech transcript in translated English format
task = 'transcribe'
 

options = dict(language=language, beam_size=5, best_of=5)
transcribe_options = dict(task=task, **options)

# Transcribe audio
device = 'cpu' # use device = cuda/gpu/mps etc if available
with torch.device(device):
    print('*** Transcribing audio file: ' + audio_file)
    result = model.transcribe(os.path.join(audio_dir, f'{audio_file}.wav'), **transcribe_options)

# Supports writing in 2 formats: SRT (subtitle) and JSON format

srt_writer = get_writer("srt", audio_dir)
srt_writer(result, f'{audio_file}.srt')

json_writer = get_writer("json", audio_dir)
json_writer(result, f'{audio_file}.json')

