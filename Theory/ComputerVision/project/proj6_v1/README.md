# Project 6: Scene Recognition with Deep Learning

## Setup

Take note that some of the concepts used in this project have **NOT** been covered in lectures, hence you may want to use this instruction page as the reference material when you proceed to each section.

We will be installing a **NEW** environment for this project; follow the instructions below to set up the env. If you run into import module errors, try “pip install -e .” again, and if that still doesn’t work, you may have to create a fresh environment.

Note that although we are training a neural net from scratch for this project, your laptop should be sufficient to handle this (expecting a 5 to 10 minutes training time for Part 1 and 2, and roughly 20 to 30 minutes for Part 3 with only the CPU); you are free to use Google Colab on this, but you may need to figure out a way of putting both the notebook and the dataset into your Google Drive and mount it in the Colab notebook (this [tutorial](https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/) covers everything you need to know to set it up).


1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyway.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj6_env_<OS>.yml`
3. This should create an environment named 'proj6'. Activate it using the Windows command, `activate proj6` or the MacOS / Linux command, `source activate proj6`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook proj6.ipynb`
6. Ensure that all sanity checks are passing by running `pytest` inside the "proj6_unit_tests/" folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` and submit to Canvas (don't forget to submit your report to Gradescope!).
