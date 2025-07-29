Run: huggingface-cli login
Run: pip install -r requirements.txt 
Run: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist 
pip install -e .



from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/gemma-3-1b-it",local_dir="models/llm/gemma-3-1b-it")

python -m spacy download en_core_web_sm