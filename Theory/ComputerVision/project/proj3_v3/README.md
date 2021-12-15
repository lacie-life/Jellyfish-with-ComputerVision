# CS 4476 project 3: [Camera Projection Matrix and Fundamental Matrix Estimation with RANSAC](https://dellaert.github.io/19F-4476/proj3.html/)


# Setup
- Install <a href="https://conda.io/miniconda.html">Miniconda</a>. It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment, using the appropriate command. On Windows, open the installed "Conda prompt" to run this command. On MacOS and Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac', or 'win'): `conda env create -f proj3_env_<OS>.yml`
- This should create an environment named `proj3`. Activate it using the following Windows command: `activate proj3` or the following MacOS / Linux command: `source activate proj3`.
- Run `pip install -e .`
- Run the notebook using: `jupyter notebook`
- Generate the submission once you're finished using `python zip_submission.py`
