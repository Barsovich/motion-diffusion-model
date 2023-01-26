# sudo apt-get -y upgrade
# sudo apt-get -y install ffmpeg

sudo apt update
sudo apt-get -y upgrade
sudo apt-get -y install ffmpeg

conda create -n mdm -y
conda init
source /home/ubuntu/.bashrc
conda activate mdm

sudo apt-get install python3-pip -y
pip install -U pip setuptools wheel
pip install -U spacy

python -m spacy download en_core_web_sm

pip install git+https://github.com/openai/CLIP.git


conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh

cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=1YWrVa1ICoW2IcDCh82EeD_qs2pkCh2nk
unzip Trinity.zip
cd ..
