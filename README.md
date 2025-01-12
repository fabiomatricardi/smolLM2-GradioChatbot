# smolLM2-GradioChatbot
a Gradio Chatbot with llama-server backhand

We will use pre-compiled llamaCPP binaries to run the model as an OpenAI compatible API server. Python and Gradio to build the Graphic interface.


<img src='https://github.com/fabiomatricardi/smolLM2-GradioChatbot/raw/main/SmolLM2-360M_gradio.gif' width=900>


I tested the application on a Windows 11 computer, with Python 3.12. The llama.cpp binaries are at version [b4464](https://github.com/ggerganov/llama.cpp/releases/tag/b4464) now, and we will use the more inclusive version wuth the AVX2 extended instructions (almost everyone can use them). You can download the [ZIP archive llama-b4464-bin-win-avx2-x64.zip directly from here](https://github.com/ggerganov/llama.cpp/releases/download/b4464/llama-b4464-bin-win-avx2-x64.zip).

> NOTE:if you clone the repo, the ZIP archive is already in the `llamacpp` directory

```
smolLM2-GradioChatbot
└───llamacpp   <--extract here the llama.cpp ZIP archive
    └───model  <--download here the GGUF file
```

The weights quantized of SmolLM2-360M-Instruct-GGUF can be downloaded from the [MaziyarPanahi HF repository](https://huggingface.co/MaziyarPanahi/SmolLM2-360M-Instruct-GGUF)

I used the Q8 version, to have almost to quality loss

Download SmolLM2-360M-Instruct.Q8_0.gguf from [here](https://huggingface.co/MaziyarPanahi/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct.Q8_0.gguf)

and put it in the subfolder `llamacpp/model`

## Venv and packages
Create a venv in the main project directory

If you clone the repo it will be called `smolLM2-GradioChatbot`

from the terminal
```
# create the virtual environment (venv)
python -m venv venv
# Activate the venv
.\venv\Scripts\activate
# intall the python packages with pip
pip install --upgrade gradio tiktoken openai
```

There is also a requirements file in the repo (but I don't reccomend it)
```
# create the virtual environment (venv)
python -m venv venv
# Activate the venv
.\venv\Scripts\activate
# intall the python packages with pip
pip install -r requirements.txt
```

### A 2 windows strategy
Gradio is our GUI creator, arrived now at version 5: it is easy and intuitive, created with Machine Learning and Generative AI applications in mind. openai is a python package to simplify the API calls to any server following the OpenAI API endpoint specifications. Finally tiktoken for some KPI to display on the app (speed, number of tokens etc…)

We apply the same concept: 

<img src='https://github.com/fabiomatricardi/smolLM2-GradioChatbot/raw/main/gradioWebApp.png' width=400><img src='https://github.com/fabiomatricardi/smolLM2-GradioChatbot/raw/main/2windowsStrategy.png' width=400>


In this project we are separating (isolating) the LLM (running as a server, compatible with the OpenAI API standard) and the application.

First we run the llama-server from the llamacpp directory: for this operation you don’t need a terminal window with venv activated (as you can see from the screenshot above)
```
.\llama-server.exe -m model\SmolLM2-360M-Instruct.Q8_0.gguf -c 8192 -ngl 0
```
In another terminal window, with the venv activated in the main project directory, simply run:
```
python .\BLgradioSmolLM2-360.py
```

> NOTE that the icons for user/chatbot are in the github repo: the app will not run if you donw't download them. If you cloned the repo... no issues!






