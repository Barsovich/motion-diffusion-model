sudo apt update
sudo apt-get -y upgrade
sudo apt-get -y install ffmpeg

conda activate pytorch
pip install -U spacy
python -m spacy download en_core_web_sm
pip install -r requirements.txt 
pip install git+https://github.com/openai/CLIP.git


bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh

cd dataset
mkdir genea_2022
mkdir genea_2022/trn
cd ..
