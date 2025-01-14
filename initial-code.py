import gradio as gr
from openai import OpenAI
import random, string, datetime
import tiktoken

#model from https://huggingface.co/MaziyarPanahi/SmolLM2-360M-Instruct-GGUF
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
# Global variables
STOPS = ['<|im_end|>'] #'<|end_of_text|>' for granite
MODELNAME = 'SmolLM2-360M-Instruct-GGUF'

# Setting up the gradio theme, with color and special Google Font - Optional
theme=gr.themes.Default(primary_hue="blue", secondary_hue="pink",
                        font=[gr.themes.GoogleFont("Oxanium"), "Arial", "sans-serif"]) 

# Main gradio Block start
with gr.Blocks(fill_width=True,theme=theme) as demo:
    # INTERFACE
    gr.Markdown("# The tite here")
    gr.Markdown("this is only a test")
    # A row as Main Container
    with gr.Row():
        # the left and right panel
        with gr.Column(scale=1):
            maxTokens = gr.Slider(minimum=100, maximum=2000, 
                                value=512, step=4, label="Max new tokens")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages",show_copy_button = True,
                                avatar_images=['user.png','bot.png'],
                                height=450, layout='bubble')      


# launch the app
demo.launch()