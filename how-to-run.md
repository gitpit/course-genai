How to run this course work in Visual Studio Code. Follow Microsoft VS Code instructions to know about initial setup etc.
https://code.visualstudio.com/docs/python/python-tutorial

A>> ## Initial Setup for running python programs##
-----------------------------------------------
1> After installing VS Code, install Python-related extensions mentioned in the tutorial link above

2> Install Python version 3.11.9. This version works best for this work. Also select pip and other tools for installation.
https://www.python.org/downloads/release/python-3119/

    Install pip
    -----------
    python -m pip install
    python.exe -m pip install --upgrade pip


3> Set up environment variables in PATH for both Python and Scripts (pip).
    C:\Program Files\Python3.11\Scripts\
    C:\Program Files\Python3.11\
3> Create a virtual environment for this project. Each project should have one venv and its dependencies.
 About: virtualenv is a tool that allows you to create virtual environments in Python and manage Python packages. It helps you avoid installing packages globally; global installations can result in breaking some system tools or other packages.
    https://learnpython.com/blog/how-to-use-virtualenv-python/
    Steps:
    1> Install virtualenv   
        pip install virtualenv

    2> Create Virtual Env for a specific Python version
        virtualenv -p python3.11 .venv3.11
            or
        py -3.11 -m venv .venv3.11
            or 
        Use Python VS Code; Ctrl + Shift + P | Python: Create environment

    3> Activate the Python virtual environment
        .\.venv3.11\Scripts\activate

    4> Check that you are in your Python virtual environment
        where python

4> Create a .env file and list all the required API Keys. Must add this file to .gitignore if not there.
 **Note: (Very important) If using any code agent (ChatGPT or CodeGPT), configure them not to read/show this file.
    OPENAI_API_KEY
    GROQ_API_KEY

5> Create a Python Requirements File to list all the installation packages in one place. These packages are installed inside the .venv* folder path.
About: Python requirements files are a great way to keep track of the Python modules. It is a simple text file that saves a list of the modules and packages required by your project. By creating a Python requirements.txt file, you save yourself the hassle of having to track down and install all of the required modules manually.
https://learnpython.com/blog/python-requirements-file/
    - First install all the packages
    - If the program works fine then add these packages to the requirements.txt file:
        
        pip freeze > requirements.txt   # this adds all the packages with version info. This holds up the dependency among the packages and avoids breaking.
    
    - Later use it to update packages:
        
        pip install -U -r requirements.txt

    - Some helpful pip commands
        pip list --outdated    # check outdated versions first
        python -m pip check    # check for missing dependencies
    pip check  # always do to check any broken libraries in requirements

    << Alternative (optional) --- using pip-tools to manage requirements file as an alternative >>
        i> Install pip-tools
            pip install pip-tools
        ii> Create requirements.in file and paste basic packages as below:
            langchain
            openai
            transformers
            streamlit
        iii> Compile this file to make requirements.txt file
            pip-compile requirements.in
        iv> Install dependencies
            pip install -r requirements.txt
        v> Install cmake and pyarrow if above fails
            https://cmake.org/download/ -- download and install

        vi> This will also require downloading VS build tools
            https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file

        vii> Install pipdeptree tool - if there is an issue and you need to check dependencies
            # activate your venv first: .\.venv\Scripts\Activate.ps1
            python -m pip install pipdeptree
        Note*** > check about wheel installation 
            python -m pip install --upgrade pip setuptools wheel
            pipdeptree --reverse --packages pyarrow

            pip install pyarrow==15.0.0 -r requirements.txt
        viii> Purge any temp pip cache
            python -m pip cache purge
            # delete any temporary pip-build folders created in %TEMP% if necessary
            python -m pip install -r requirements.txt
        
        ix> If some package listed in requirements.txt file has dependency issues; then
            install it explicitly outside of the requirements file
                python -m pip install pyarrow==15.0.0

        x> Modify .gitignore to remove any clutter
            venv/
            __pycache__/
            *.pyc
        xi> When a new package has to be added - always add to Requirements.in File
            # Add new package
            echo "flask" >> requirements.in
            # Compile pinned requirements
            pip-compile requirements.in
            # Install them
            pip install -r requirements.txt
            # Upgrade all packages (optional)
            pip-compile --upgrade requirements.in
            # Sync environment (optional)
            pip-sync requirements.txt  # pip-sync cleans up your environment so it only contains what your project needs.
                                        # This avoids dependency conflicts and keeps things reproducible.
                                        # pip-sync adds, updates, AND removes anything not in the requirements file.



6> Use normal debugger commands to debug and run programs...
---------------------------------------------------------------------------------------------

B>> ## Instructions for Python code debugging using pdb ##
    1> Install pdbpp
        pip install pdbpp
    2> Set up .pdbrc.py file to keep basic configuration on load
    3> Start debug 
    python -m pdb transformer_scratch.py
    4> Some useful commands for debug
        Some useful ones to remember are:
            b: set a breakpoint
            c: continue debugging until you hit a breakpoint
            s: step through the code
            n: go to next line of code
            l: list source code for the current file (default: 11 lines including the line being executed)
            u: navigate up a stack frame
            d: navigate down a stack frame
            p: print the value of an expression in the current context


C>> ### Ollama setup for running models locally and using for inference/retrieval ###
    <<Steps:>>
    ## Install Ollama:
        Download and install from https://ollama.com/download.
    ## Start Ollama:
        Open a terminal and run:
        ollama serve
    ## Pull a model:
        For example: 
        ollama pull llama2
    ## Install Python client:
        pip install ollama
    
    <<Download models in custom directory:>>
    First, stop the Ollama service if running:
    ## Stop Ollama service
        ollama stop
    # Start Ollama with custom path
        ollama serve --models-path "D:\llm_models\ollama"
    # In another terminal, pull models
        ollama pull gemma:2b
    from langchain.llms import Ollama

    # Initialize Ollama with custom path
    llm = Ollama(
        model="gemma:2b",
        base_url="http://localhost:11434",
        models_path=r"D:\llm_models\ollama"
    
    <<Troubleshooting:>>
    # Check if Ollama is already running (it may be running in the background).
    # Find and stop the process using port ex: 11434: Note the PID (last column).
        netstat -ano | findstr :11434
        
        taskkill /F /PID <pid>
    # Try running ollama serve again (ollama serve)
    # To set a specific folder to download and install model with Ollama, set this
    set OLLAMA_MODELS=E:\your\desired\directory
    >ollama pull llama2
    # What should be the Order?:
    1> Start the server:
    >ollama serve
    2> In another terminal (or the same one), pull the model:
    > ollama pull llama2
    This ensures the server is running and ready to manage models.

    # Pull a specific model + tag (use exact model name)
    ollama pull embeddinggemma:300m
    # Verify downloaded models
    ollama list
    # Start the Ollama server (if not already running)
    ollama serve
    # See local models folder (default)
    dir $env:LOCALAPPDATA\ollama
    # Set custom models dir before pulling (PowerShell)
    $env:OLLAMA_MODELS = "D:\llm_models\ollama"
    ollama pull embeddinggemma:300m


D> ## For the case of Hugging Face - steps to download model in specific folder ##
    model = SentenceTransformer(model_name) - downloads model locally
    Changing SentenceTransformer Cache Location to specific folder
    # Load model with custom cache location for **transformer embeddings**
    # Define custom cache directory
    custom_cache_dir = r"D:\models\embeddings"  # Change this to your desired path
    os.makedirs(custom_cache_dir, exist_ok=True)
    model = SentenceTransformer(
        model_name,
        cache_folder=custom_cache_dir
    ).to(device)

    Alternatively set env path
    os.environ['TORCH_HOME'] = r"D:\models\embeddings"

    # Load model with custom cache location for **transformer models **
    # Set custom cache directory before creating pipeline
    os.environ['TRANSFORMERS_CACHE'] = r'D:\models\gemma'
    custom_cache_dir = r"D:\models\gemma"  # Change this to your desired path

    pipeline = pipeline(
        task="text-generation",
        model="google/gemma-2b-it",
        model_kwargs={
            "cache_dir": custom_cache_dir
        },
        device_map="auto",
        torch_dtype=torch.float16    
    )