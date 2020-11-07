# Create your conda env

Kaggle say:

> Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.

So these I have installed (conda install pytorch, pip install gym, rest come along with those requirements)

## 1. Get anaconda

### General instructions
https://docs.anaconda.com/anaconda/install/


### Ubuntu Linux shortcut

Follow along here https://docs.anaconda.com/anaconda/install/linux/

#### 1. Download 

https://www.anaconda.com/products/individual#linux

#### 2. Recommended verify file integrity 

It's recommended to check you have the right version for your operating system. There is a tool that checks the file hash that you're about to download against the expected version's hash, which you get from this page: https://docs.anaconda.com/anaconda/install/hashes/

Specifically for ubuntu linux, 64bit, anaconda3 (do check your OS yourself): https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/

Get the relevant file hash here
`sha256sum /path/filename`

Where /path/filename is the relevant hash you just downloaded

#### 3. Install the downloaded file

`bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh`

## 2. Create an env from yaml

Check you now have conda

`which conda`

Then you can install an environment from this yaml file

`conda -f conda_env.yml`

`conda activate common`

As common is the name of the environment, as-per the file.

To deactivate again

`conda deactivate` 

Read the docs for more...
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
