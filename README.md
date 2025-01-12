# smolLM2-GradioChatbot
a Gradio Chatbot with llama-server backhand

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

There is also a requireements file in the repo (but I don't reccomend it
```
# create the virtual environment (venv)
python -m venv venv
# Activate the venv
.\venv\Scripts\activate
# intall the python packages with pip
pip install -r requirements.txt
```




