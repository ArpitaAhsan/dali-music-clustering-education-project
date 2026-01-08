# DALI Music Clustering Project 
 
Project implements unsupervised music genre clustering using VAEs on DALI dataset. 
3363 tracks across 72 genres. Best result: AE+DBSCAN Silhouette=0.680. 
 
## Dataset 
 
Audio (3363 tracks, 30s clips): https://drive.google.com/drive/folders/1Va7zXP6ZB731QAPkeu-iSSmu_BoPZvpd?usp=sharing 
Metadata (fma_style_dali_6000.csv): https://drive.google.com/file/d/124cahFkyjAOkHIV0plmBZ6kYJLySnO9r/view?usp=sharing 
 
CSV columns: title, artist, genre_top, transcript, filepath, duration 
BASE_PATH: /content/drive/MyDrive/425_DALI/ 
Audio example: /content/drive/MyDrive/425_DALI/audio/001940b614eb43f4a0c826d49a67d66d.wav 
 
## Setup 
pip install -r requirements.txt 
 
## Tasks 
- notebooks/exploratory.ipynb: Easy/Medium/Hard tasks 
- src/vae.py: VAE implementations 
- src/dataset.py: Data loading 
- src/clustering.py: Clustering algorithms 
