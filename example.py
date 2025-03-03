from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag import *


# configs for embedding model
embedding_model = "iic/nlp_gte_sentence-embedding_chinese-small"

# file path for your custom knowledge base
knowledge_doc_file_dir = "/mnt/workspace/RAG/custom_data/"


qwllm = QianWenChatLLM()
print('STEP1: qianwen LLM created')

# STEP2: load knowledge file and initialize vector db by llamaIndex
print('STEP2: reading docs ...')
embeddings = ModelScopeEmbeddings4LlamaIndex(model_id=embedding_model)
Settings.llm = None
Settings.embed_model=embeddings  # global config, not good

llamaIndex_docs = SimpleDirectoryReader(knowledge_doc_file_dir).load_data()
llamaIndex_index = VectorStoreIndex.from_documents(llamaIndex_docs, chunk_size=4096)
retriever = LlamaIndexRetriever(index=llamaIndex_index)
print(' 2.2 reading doc done, vec db created.')

# STEP3: create chat template
prompt_template = """请用中文回答，请基于```内的内容回答问题。"
```
{context}
```
我的问题是：{question}。
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)
print('STEP3: chat prompt template created.')

# STEP4: create RAG chain to do QA
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | qwllm
        | StrOutputParser()
)

# question = 'What kinds of attention mechanisms are mentioned in the text'
question = 'What kinds of attention mechanisms are mentioned in the text'
# chain.invoke(question)
answer = chain.invoke(question)
print(answer)