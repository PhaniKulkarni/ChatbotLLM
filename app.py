import os

from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader


from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()
system_prompt = """<|SYSTEM|># SartajLM Tuned
- SartajLM is a helpful and harmless open-source AI language model developed by Sartaj.
- SartajLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- SartajLM's primary goal is to answer questions based on the custom data that is fed to it.
- It would prioritize knowledge from additional files it has receved.
- SartajLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

import torch
from llama_index.llms import HuggingFaceLLM


from flask import Flask, render_template, request
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="facebook/opt-iml-max-1.3b",
    model_name="facebook/opt-iml-max-1.3b",
    device_map="auto",
    #offload_folder="",
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)


app: Flask = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        query_engine = index.as_chat_engine()
        response = query_engine.query(query)
        return render_template('index.html', query=query, response=response)
    return render_template('index.html', query=None, response=None)

if __name__ == '__main__':
    app.run()