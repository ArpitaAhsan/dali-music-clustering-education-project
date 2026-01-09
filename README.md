# DALI Music Clustering Project 
 
Project implements unsupervised music genre clustering using VAEs on DALI dataset. 
3363 tracks across 72 genres. Best result: AE+DBSCAN Silhouette=0.680. 
 
## Dataset 
 
Audio (3363 tracks, 30s clips): This hsa all the audio files.
https://drive.google.com/drive/folders/1Va7zXP6ZB731QAPkeu-iSSmu_BoPZvpd?usp=sharing 

Metadata (fma_style_dali_6000.csv): For each track it has a row containing title, artist, genre_top, transcript, filepath, duration. The filepath has the location of the correspnding audio file
https://drive.google.com/file/d/124cahFkyjAOkHIV0plmBZ6kYJLySnO9r/view?usp=sharing 
 
CSV columns: title, artist, genre_top, transcript, filepath, duration 

BASE_PATH: /content/drive/MyDrive/425_DALI/ 

Audio example: /content/drive/MyDrive/425_DALI/audio/001940b614eb43f4a0c826d49a67d66d.wav 
 
## Tasks 
All the tasks (Easy, Medium ,Hard) are done inside a colab file that has seperate cells for each task. The link to that colab file is given inside the notebooks/README.md. 
- notebooks/README.md: Easy/Medium/Hard tasks 
- src/vae.py: VAE implementations 
- src/dataset.py: Data loading 
- src/clustering.py: Clustering algorithms

## Setup 
pip install -r requirements.txt 
