#!/bin/bash

actionspot=$1  # Get value from the first argument

sudo apt install -y p7zip-full

# Setup virtual env
python3 -m venv myenv
source myenv/bin/activate

# Install necessary packages
pip install -r sn-teamspotting/requirements.txt
pip install huggingface_hub[cli]
pip install SoccerNet --upgrade
pip install -q wheel transformers accelerate datasets peft bitsandbytes tensorboard av num2words ipywidgets tf-keras
pip install -q flash-attn 
pip install -q -U pillow jinja2
pip install numpy==1.26.0

# Run Python script for data download
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='SoccerNet/SN-BAS-2025',
                  repo_type='dataset', revision='main',
                  local_dir='SoccerNet/SN-BAS-2025')
"

# Optionally download the action spotting data
if [ "$actionspot" == "true" ]; then
    echo "Action spot is enabled, will proceed to download videos"
    python3 -c "
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory='~/SoccerNet')
mySoccerNetDownloader.password = 's0cc3rn3t'
mySoccerNetDownloader.downloadGames(files=['1_720p.mkv', '2_720p.mkv'], split=['train','valid','test','challenge'])
mySoccerNetDownloader.downloadGames(files=['1_224p.mkv', '2_224p.mkv'], split=['train','valid','test','challenge'])
"
else
    echo "Action spot is disabled, will not download videos"
fi

cd SoccerNet/SN-BAS-2025

# Unzip videos and move into videos folder
7z x -ps0cc3rn3t test.zip
7z x -ps0cc3rn3t train.zip
7z x -ps0cc3rn3t valid.zip
7z x -ps0cc3rn3t challenge.zip
mkdir -p videos
mv england_efl videos

# Unzip extra labels
if [ "$actionspot" == "true" ]; then
    cd ExtraLabelsActionSpotting500games
    unzip train_labels.zip
    unzip test_labels.zip
    unzip valid_labels.zip
else
    echo "Action spot is disabled, will not unzip extra labels"
fi

# Return to home directory before going to sn-teamspotting
cd ~

cd sn-teamspotting

python3 extract_frames_snb.py --video_dir ~/SoccerNet/SN-BAS-2025/videos --out_dir ~/frames_snb --sample_fps 25 --num_workers 5

if [ "$actionspot" == "true" ]; then
    echo "Action spot is enabled, will proceed to extract frames"
    python3 extract_frames_sn.py --video_dir ~/SoccerNet/SN-BAS-2025/ --out_dir ~/frames_sn --sample_fps 12.5 --num_workers 5
else
    echo "Action spot is disabled, will not extract frames"
fi