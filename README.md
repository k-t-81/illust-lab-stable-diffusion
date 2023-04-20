main stack
-----
- runtime
  - python 3.10.9
  - cuda toolkit 11.8
- server 
  - fast-api
  - unicorn
- machine-learning
  - diffusers

start server
------------
`source venv/bin/activate`  
`python main.py`  

install python 3.10.9
---------------------
`pyenv install 3.10.9`  
`pyenv global 3.10.9`  
`python -m venv venv`  
`source venv/bin/activate`  

install cuda toolkit 11.8
-------------------------

wsl2 ubuntu 20.04  (windows 11)  
`wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin`  
`sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600`  
`wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb`  
`sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb`  
`sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/`  
`sudo apt-get update`  
`sudo apt-get -y install cuda`
  
`export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`  

install packages
----------------
`source venv/bin/activate`  
`pip install --no-cache-dir torch==2.1.0.dev20230327+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117`  
`pip install --no-cache torchvision==0.16.0.dev20230327+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117`  
`pip install -r requirements.txt`

compress model
-------------
`python pipeline.py`  

colab
-----
`!git config --global user.name "username"`  
`!git config --global user.emial "email"`    
  
`from google.colab import drive`  
`drive.mount("./drive")`  
  
`!mkdir -p drive/MyDrive/.ssh`  
`!ssh-keygen -t rsa -f drive/MyDrive/.ssh/id_rsa_github`  
  
`!cat drive/MyDrive/.ssh/id_rsa_github.pub`  
  
`!mkdir /root/.ssh`  
`!chmod 600 /root/.ssh`  
`!cp drive/MyDrive/.ssh/id_rsa_github /root/.ssh/id_rsa`  
`!cp drive/MyDrive/.ssh/id_rsa_github.pub /root/.ssh/id_rsa.pub`  
`!ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts`  
  
`!git clone git@github.com:k-t-81/illust-lab-stable-diffusion.git`  

`import shutil`  
`source_path = "drive/MyDrive/pipelines"`  
`destination_path = "illust-lab-stable-diffusion/pipelines"`
`shutil.copytree(source_path, destination_path)`
  
`!nvcc -V`  

`%cd illust-lab-stable-diffusion/`  
  
`!ls`  
`!pip install --no-cache-dir torch==2.1.0.dev20230327+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117`  
`!pip install --no-cache torchvision==0.16.0.dev20230327+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117`  
`!pip install -r ./requirements.txt`  
  
`python pipeline.py`  
  
`import shutil`  
`source_path = "drive/MyDrive/pipelines"`  
`destination_path = "illust-lab-stable-diffusion/pipelines"`
`shutil.copytree(destination_path, source_path)`  
  
`!ngrok authtoken token`  

`!python ./colab.py`  

more info
---------
- fast-api: https://fastapi.tiangolo.com/ja/


- cuda: https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-cuda-on-wsl2 


- diffusers: https://huggingface.co/docs/diffusers/installation  
- diffusers with torch2.0: https://huggingface.co/docs/diffusers/optimization/torch2.0


- txt2img: https://huggingface.co/docs/diffusers/v0.15.0/en/api/pipelines/stable_diffusion/text2img
- img2img: https://huggingface.co/docs/diffusers/v0.15.0/en/api/pipelines/stable_diffusion/img2img
- controlnet: https://huggingface.co/docs/diffusers/v0.15.0/en/api/pipelines/stable_diffusion/controlnet


- huggingface model Counterfeit-v2.5: https://huggingface.co/gsdf/Counterfeit-V2.5  
- huggingface model anything-v3.0: https://huggingface.co/Linaqruf/anything-v3.0  
- huggingface model sd-controlnet-canny: https://huggingface.co/lllyasviel/sd-controlnet-canny