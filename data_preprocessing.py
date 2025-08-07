# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import re
import langid

#%% loading our data

df = pd.read_csv('dataset\labeled_lyrics_cleaned.csv')

#%% removing non-english songs

'''
Although the data was cleaned from non-english songs by ASCII letter
we still got songs in spanish, german, french and etc.
We will no treat them differently
'''
bad_inputs = []

langs = []
bad_inputs = []
total = len(df)

for i, text in enumerate(df['seq']):
    # Progress print every 50 samples
    if i % 10000 == 0:
        print(f"Processing {i}/{total} samples...")

    # Handle invalid or empty inputs
    if not isinstance(text, str) or not text.strip():
        langs.append('invalid')
        bad_inputs.append((i, text))
        continue
    
    # Language detection with error handling
    try:
        lang = langid.classify(text)[0]
        langs.append(lang)
    except Exception as e:
        langs.append('error')
        bad_inputs.append((i, text))

#%%

# Save results back to DataFrame
df['lang'] = langs
df = df.drop(columns=['Unnamed: 0'])
df_en = df[df['lang'] == 'en']
df_en =  df_en.drop(columns=['lang'])
#%%
# Normalize line endings in 'seq' column
df_en = pd.read_csv('dataset\english_lyrics.csv')
df_en['lyrics'] = df_en['lyrics'].apply(lambda x: x.replace('\r\n', '\n').replace('\r', '\n'))
df_en.to_csv("english_lyrics_new.csv", index=False)

#%%

# =============================================================================
# def normalize_linebreaks(text):
#     text = re.sub(r'\n+', '\n', text)        # collapse multiple line breaks
#     text = re.sub(r'[ ]{2,}', ' ', text)     # collapse multiple spaces
#     return text.strip()
# =============================================================================



















