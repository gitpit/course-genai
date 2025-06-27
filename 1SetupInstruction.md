Install Python
https://code.visualstudio.com/docs/python/python-tutorial

Install required Python extensions from vscode extensions

Setup environment (as below or from cmd or pyshell)
Install pip
python -m pip install
python.exe -m pip install --upgrade pip

A>> Create virtual environment in VSCode 
    https://learnpython.com/blog/how-to-use-virtualenv-python/

    About VirtualEnv:
    virtualenv is a tool that allows you to create virtual environments in Python and manage Python packages. It helps you avoid installing packages globally; global installations can result in breaking some system tools or other packages.

    Steps:
    1> Installation pip
    pip install --upgrade pip
    pip --version
    pip install virtualenv

    2> Create Virtual Env
    virtualenv -p python3 mytest

    3> Python virtual environment, you need to activate it. 
    .\mytest\Scripts\activate

    4> Check that you are in your Python virtual environment
    where python

B>> Create a Python Requirements File:
https://learnpython.com/blog/python-requirements-file/
Python requirements files are a great way to keep track of the Python modules. It is a simple text file that saves a list of the modules and packages required by your project. By creating a Python requirements.txt file, you save yourself the hassle of having to track down and install all of the required modules manually.
    pip list --outdated    #check outdated version first
    pip install -U <packagename>
    pip freeze > requirements.txt  #to create and update the Python requirements file.
    pip install -U -r requirements.txt  #It is also possible to upgrade everything with 
    python -m pip check    #check for missing dependencies
    pip check  #always do to check any broken lib in requirment

C>> For cohort GenAI course setup
    1> pip install groq dotenv gradio
    2> Get Groq's API key from https://console.groq.com/keys
    3> test groq with a simple prompt

c> Python code debugging using pdb
    1> install pdbpp
        pip insall pdbpp
    2> setup .pdbrc.py file to keep basic configuration on load
    3> Some useful commands for debug
        Some useful ones to remember are:
            b: set a breakpoint
            c: continue debugging until you hit a breakpoint
            s: step through the code
            n: to go to next line of code
            l: list source code for the current file (default: 11 lines including the line being executed)
            u: navigate up a stack frame
            d: navigate down a stack frame
            p: to print the value of an expression in the current context