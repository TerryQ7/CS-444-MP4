### Suggestions for setup, workflow, and FAQs

#### Development environment
1. You should develop and debug locally, rather than trying to develop directly
   against gradescope autograding. Once you are able to get tests to pass locally,
   only then should you upload to gradescope for grading. Directly debugging
   against gradescope will be quite frustrating as a) it will be slow, and b)
   you will not have access to the full error messages.

2. If you have python already installed on your system, we recommend using
[virtualenv](https://docs.python.org/3/library/venv.html) to create a
separate isolated environment (i.e., libraries and packages) for use with
this assignment. 
    1. Create a virtualenv: `python3 -m venv <name_of_virtualenv>`
    2. Start using the environment: `source <name_of_virtualend>/bin/activate`.
    3. You can then install the required libraries within the virtualenv using: `pip install -r requirements.txt`.
    4. You should now be able to run the `demo.py` script in your terminal.
    
    This creates a new directory `name_of_virtualenv` and installs all the
    required libraries within this folder. So, these remain separate from
    other python libraries that you may have installed on your system.

    If you don't have python installed on your system, first install python and then follow the above steps. 

3. Alternatively, you can also setup your environment using
[Conda](https://docs.conda.io/en/latest/). Conda takes isolation a step
further and also installs a separate python.
    1. First, install conda by going to this
    [link](https://conda.io/projects/conda/en/stable/user-guide/install/download.html)
    and installing miniconda. You can check if conda is installed by typing
    in conda in the terminal.
    2. Use [environment.yml](./environment.yml) to create a conda
    environment via: `conda env create -f environment.yml`. This will follow
    the instructions in the [environment.yml](./environment.yml) file by
    creating a python virtual environment, installing python and then
    installing all the required packages in the `requirements.txt` file with
    pip.
    3. Then, activate your conda environment by running `conda activate
    assignment1`. This will activate the conda environment named
    `assignment1`.  Afterwards your terminal should look as follows:
        ```
        (assignment1) saurabhg@saurabhg-desktop$
        ```
    4. You should now be able to run the `demo.py` script in your terminal.

4. If you end up having multiple different python environments, you can run 
   into the issue of not knowing a) what environment you are currently 
   using, or b) what environment are packages getting installed into. Running
   something like `python3 -c 'import absl; print(absl.__file__)` can help
   understand what environment you are currently using.

5.  **Jupyter Notebooks.** We have only provided the code in the form of python
scripts and functions in `.py` files. Depending on your working style, you may
benefit from interactively developing your code in a [Jupyter
Notebook](https://jupyter.org/). This lets you iterate on smaller chunks of code
and also visualize the results as you go. You can then copy the code into python
scripts for running longer trainings and for submission. 
    - Jupyter Notebooks are particularly useful for visualizing intermediate
    results which may be quite useful for problems 3 and 4. 
    - You may also find profiling `line_profiler` useful for optimizing your
    code (can also be easily used from within Jupyter Notebooks). 
    - You may also benefit from using
    [autoreload](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html)
    if you change a file and want to reload it without restarting the kernel.