from abc import ABC
from typing import Any, List, Optional, Dict, cast

from langchain_core.language_models.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import ollama
import subprocess


def ollama_init():
    commands = "export OLLAMA_MODELS=/mnt/workspace/.cache/ollama/model;ollama serve"
    process = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# define our Embedding class to use models in Modelscope
class ModelScopeEmbeddings4LlamaIndex(BaseEmbedding, ABC):
    embed: Any = None
    model_id: str = "iic/nlp_gte_sentence-embedding_english-large"

    def __init__(
            self,
            model_id: str,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from modelscope.models import Model
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            self.embed = pipeline(Tasks.sentence_embedding, model=self.model_id)

        except ImportError as e:
            raise ValueError(
                "Could not import some python packages." "Please install it with `pip install modelscope`."
            ) from e

    def _get_query_embedding(self, query: str) -> List[float]:
        text = query.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        return self.embed(input=inputs)['text_embedding'][0]

    def _get_text_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        return self.embed(input=inputs)['text_embedding'][0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        inputs = {"source_sentence": texts}
        return self.embed(input=inputs)['text_embedding']

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


# define our Retriever with llama-index to co-operate with Langchain
# note that the 'LlamaIndexRetriever' defined in langchain-community.retrievers.llama_index.py
# is no longer compatible with llamaIndex code right now.
class LlamaIndexRetriever(BaseRetriever):
    index: Any
    """LlamaIndex index to query."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.core.indices.base import BaseIndex
            from llama_index.core import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseIndex, self.index)
        print('@@@ query=', query)

        response = index.as_query_engine().query(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            print('@@@@ source=', source_node)
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.get_text(), metadata=metadata)
            )
        return docs

# define QianWen LLM based on langchain's LLM to use models in Modelscope
class QianWenChatLLM(LLM):
    llm_name: str = "qwen2.5:7b-8k"
    def __init__(self):
        super().__init__()
        ollama_init()

    @property
    def _llm_type(self):
        return "ChatLLM"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs: Any,
    ) -> str:
        #print(prompt)

        response  = ollama.generate(
            model=self.llm_name,
            prompt= prompt,
        )
        #print(response.response)
        return response.response