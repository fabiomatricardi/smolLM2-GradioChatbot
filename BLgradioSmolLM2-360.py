import time
import gradio as gr
from openai import OpenAI
import random
import string
import tiktoken
import datetime
#https://huggingface.co/MaziyarPanahi/SmolLM2-360M-Instruct-GGUF

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
STOPS = ['<|im_end|>'] #'<|end_of_text|>' for granite
MODELNAME = 'SmolLM2-360M-Instruct-GGUF'

def countTokens(text):
    """
    Use tiktoken to count the number of tokens
    text -> str input
    Return -> int number of tokens counted
    """
    encoding = tiktoken.get_encoding("cl100k_base") 
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return f'Logfile_{res}.txt'

LOGFILENAME = genRANstring(5)

theme=gr.themes.Default(primary_hue="blue", secondary_hue="pink",
                        font=[gr.themes.GoogleFont("Oxanium"), "Arial", "sans-serif"]) 

#Inconsolata,Oxanium,Saira,IBM Plex Serif,Roboto Flex,Raleway,Source Serif 4,Slabo 27px,Fira Sans Condensed,Saira Condensed,Jura

with gr.Blocks(fill_width=True,theme=theme) as demo:
    # INTERFACE
    with gr.Row(height=35,variant='panel'):
        gr.Markdown(
        f"""### Advanced AI ChatBot Interface - Running *{MODELNAME}* with llamaCPP-server""")    
    with gr.Row():
        #HYPERPARAMETERS
        with gr.Column(scale=1):
            maxTokens = gr.Slider(minimum=100, maximum=2000, value=512, step=4, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.1, step=0.1, label="Temperature")
            repeatPenalty = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.35,
                step=0.05,
                label="Frequency Penalty",
            )
            # KPIs
            TTFT = gr.Text(label='Time to First Token',value='\n',container=False)
            INFTIME = gr.Text(label='Inference Time',value='0 seconds',)
            SPEED = gr.Text(label='Generation Speed',value='0 t/s',)
            TOKENSKPI = gr.Text(label='Tokens Stats',value='\n\n\n',container=False)
            LOGFILE = gr.Text(label='Log Filename',value=LOGFILENAME,container=False)

        #CHATBOT AREA    
        with gr.Column(scale=3):    
            chatbot = gr.Chatbot(type="messages",show_copy_button = True,
                                avatar_images=['user.png','bot.png'],
                                height=450, layout='bubble')
            msg = gr.Textbox(lines=3)
            clear = gr.ClearButton([msg, chatbot])

            def chat(message, history,t,r,m):
                firstToken = 0
                history.append({"role": "user", "content": message})
                entire_prompt = ''
                for items in history:
                    entire_prompt += items['content']                
                prompttokens = countTokens(entire_prompt)
                messagetokens = countTokens(message)
                start = datetime.datetime.now()
                stream = client.chat.completions.create(
                                messages=history, 
                                model=MODELNAME, 
                                temperature=t,
                                frequency_penalty  = r,
                                stop=STOPS,
                                max_tokens=m,
                                stream=True)
                history.append({"role": "assistant", "content": ""})
                clear_prompt = ''
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        if firstToken == 0:
                            ttft_time = datetime.datetime.now() - start
                            genStartTime = datetime.datetime.now()
                            ttft_seconds = ttft_time.total_seconds()
                            gr_ttft_seconds = f'\n'
                            history[-1]['content'] += chunk.choices[0].delta.content
                            firstToken = 1
                        else:    
                            history[-1]['content'] += chunk.choices[0].delta.content
                        assistanttokens = countTokens(history[-1]['content'])
                        TdeltaGeneration = (datetime.datetime.now() - genStartTime)
                        deltaGeneration = TdeltaGeneration.total_seconds()
                        Ttotalseconds = (datetime.datetime.now() - start)
                        totalseconds = Ttotalseconds.total_seconds()
                        assistanttokens = countTokens(history[-1]['content'])
                        totaltokens = prompttokens + assistanttokens
                        overallSpeed = totaltokens/totalseconds
                        evalSpeed = prompttokens/ttft_seconds
                        gr_ttft_seconds = f'Time to First Token: {ttft_seconds:.1f} seconds\nPrompt Eval: {evalSpeed:.2f}  t/s in {ttft_seconds:.1f} seconds'
                        try:
                            genspeed = assistanttokens/deltaGeneration  
                        except:
                            genspeed = 0  
                        gr_InferenceTime = f'{totalseconds} seconds'                                #INFTIME
                        gr_genspeed = f'{genspeed:.2f}  t/s'                                        #SPEED
                        gr_tokens = f'Prompt Tokens: {messagetokens}\nChat History Tokens: {prompttokens}\nOutput Tokens: {assistanttokens}\nTOTAL Tokens: {totaltokens}'                      
                    yield history, clear_prompt, gr_ttft_seconds, gr_InferenceTime,gr_genspeed,gr_tokens
                # Save in the log file prompt and reply
                stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⏱️ Inference time:   {totalseconds} seconds
>>>🏍️ Inference speed:  {overallSpeed:.2f}  t/s
>>>🏃‍♂️ Generation speed: {genspeed:.2f}  t/s
>>>🤔 Prompt EVAL speed: {evalSpeed:.2f}  t/s
>>>⏳ {gr_ttft_seconds}
{gr_tokens}
---
'''             
                tosave = f'{datetime.datetime.now()}\n👷USER > {message}\n🦙 BOT > {history[-1]['content']}\n---\n{stats}\n\n'
                writehistory(LOGFILENAME,tosave)    

            
            clear.click(lambda: None, None, chatbot, queue=False)

            msg.submit(chat, [msg, chatbot,temperature,repeatPenalty,maxTokens], [chatbot,msg,TTFT,INFTIME,SPEED,TOKENSKPI])


demo.launch()



###########ICL PROMPT##################################
"""
                    model="SmolLM2-360M-Instruct-GGUF", 
                    temperature=0.1,
                    frequency_penalty  = 1.35,
                    stop=STOPS,
                    max_tokens=500,

                    
Task: reply to the user question using a provided context and say "unanswerable" when the context does not have the information required.

Examples:

[context] Large language models (LLMs) excel at few-shot in-context learning (ICL) – learning from a few inputoutput examples (“shots”) provided in context at inference, without any weight updates. Newly expanded context windows allow us to investigate ICL with hundreds or thousands of examples – the many-shot regime. Going from few-shot to many-shot, we observe significant performance gains across a wide variety of generative and discriminative tasks. While promising, many-shot ICL can be bottlenecked by the available amount of human-generated outputs. To mitigate this limitation, we explore two settings: (1) “Reinforced ICL” that uses model-generated chain-of-thought rationales in place of human rationales, and (2) “Unsupervised ICL” where we remove rationales altogether, and prompt the model only with domain-specific inputs. We find that both Reinforced and Unsupervised ICL can be effective in the many-shot regime, particularly on complex reasoning tasks. Furthermore, we demonstrate that, unlike few-shot learning, many-shot learning is effective at overriding pretraining biases, can learn high-dimensional functions with numerical inputs, and performs comparably to fine-tuning.[end of context] Remember: If the answer is not contained in the text say "unanswerable" and explain why you cannot answer. question: what is Science? answer: unanswerable. The provided text discusses Large Language Models (LLMs) and their capabilities, but it does not provide any information about the definition of science.

[context] Today, I will share more details about Russia’s collapsing currency. Russia is not headed for stagflation. They are on the path towards bankruptcy. The Bank of Russia will not buy any foreign currency until the end of the year. In August 2023, there was a similar announcement. Back then, the ruble was also worth less than one cent. The Bank of Russia is the only player in this market. India and China both refuse to take payments in rubles. These developments will continue to have adverse effects on Russian food prices, energy exports, and other areas of their economy. Dictators tell you a lie, and you are supposed to believe it. “The Russian economy is like a turtle that slides down a greased slippery slide and tries to go up the creek without a paddle.” Mark Biernart. Russia’s piggy bank and especially the liquid assets of the National Wealth Fund, are slowly but steadily disappearing. Some say the collapsing ruble is increasing the revenue of Russian fuel exports. Yes, that’s true. However, this drives up Russian import costs and Russia’s biggest issue that the Kremlin lacks the labor to meet those demands. Russian imports will become more expensive, up and down the chain, and the inflation inside Russia will get much worse.[end of context] Remember: If the answer is not contained in the text say "unanswerable" and explain why you cannot answer. question: what is Bill Clinton policy? answer: unanswerable. The provided text does not contain any information about Bill Clinton's policies. It focuses on Russia's economic situation.

Now reply to the user question:
[context] Today, I will share more details about Russia’s collapsing currency. Russia is not headed for stagflation. They are on the path towards bankruptcy. The Bank of Russia will not buy any foreign currency until the end of the year. In August 2023, there was a similar announcement. Back then, the ruble was also worth less than one cent. The Bank of Russia is the only player in this market. India and China both refuse to take payments in rubles. These developments will continue to have adverse effects on Russian food prices, energy exports, and other areas of their economy. Dictators tell you a lie, and you are supposed to believe it. “The Russian economy is like a turtle that slides down a greased slippery slide and tries to go up the creek without a paddle.” Mark Biernart. Russia’s piggy bank and especially the liquid assets of the National Wealth Fund, are slowly but steadily disappearing. Some say the collapsing ruble is increasing the revenue of Russian fuel exports. Yes, that’s true. However, this drives up Russian import costs and Russia’s biggest issue that the Kremlin lacks the labor to meet those demands. Russian imports will become more expensive, up and down the chain, and the inflation inside Russia will get much worse.[end of context]
Remember: If the answer is not contained in the text say "unanswerable" and explain why you cannot answer.
question: what is Bill Clinton policy?
answer:

unanswerable. The provided text does not contain any information about Bill Clinton's policies, which are discussed in a different context of Russia's economic situation.



[context] Large language models (LLMs) excel at few-shot in-context learning (ICL) – learning from a few inputoutput examples (“shots”) provided in context at inference, without any weight updates. Newly expanded context windows allow us to investigate ICL with hundreds or thousands of examples – the many-shot regime. Going from few-shot to many-shot, we observe significant performance gains across a wide variety of generative and discriminative tasks. While promising, many-shot ICL can be bottlenecked by the available amount of human-generated outputs. To mitigate this limitation, we explore two settings: (1) “Reinforced ICL” that uses model-generated chain-of-thought rationales in place of human rationales, and (2) “Unsupervised ICL” where we remove rationales altogether, and prompt the model only with domain-specific inputs. We find that both Reinforced and Unsupervised ICL can be effective in the many-shot regime, particularly on complex reasoning tasks. Furthermore, we demonstrate that, unlike few-shot learning, many-shot learning is effective at overriding pretraining biases, can learn high-dimensional functions with numerical inputs, and performs comparably to fine-tuning.[end of context]
Remember: If the answer is not contained in the text say "unanswerable" and explain why you cannot answer.
question: what is Science?
answer:

unanswerable. The provided text does not contain any information about Science, which is discussed in a different context of human knowledge and understanding.

"""