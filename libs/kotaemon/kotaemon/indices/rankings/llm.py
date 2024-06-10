from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from decouple import config
from langchain.output_parsers.boolean import BooleanOutputParser

from kotaemon.base import Document
from kotaemon.llms import BaseLLM, PromptTemplate

from .base import BaseReranking

RERANK_PROMPT_TEMPLATE = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.
Answer very concisely in English. Avoid having both YES and NO in the answer.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""


class LLMReranking(BaseReranking):
    llm: BaseLLM
    prompt_template: PromptTemplate = PromptTemplate(template=RERANK_PROMPT_TEMPLATE)
    top_k: int = 3
    concurrent: bool = True

    def run(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []
        output_parser = BooleanOutputParser()

        if self.concurrent:
            with ThreadPoolExecutor() as executor:
                futures = []
                for doc in documents:
                    _prompt = self.prompt_template.populate(
                        question=query, context=doc.get_content()
                    )
                    futures.append(executor.submit(lambda: self.llm(_prompt).text))

                results = [future.result() for future in futures]
        else:
            results = []
            for doc in documents:
                _prompt = self.prompt_template.populate(
                    question=query, context=doc.get_content()
                )
                results.append(self.llm(_prompt).text)

        # use Boolean parser to extract relevancy output from LLM
        parsed_results = [output_parser.parse(result) for result in results]
        for idx, include_doc in enumerate(parsed_results):
            doc = documents[idx]
            if config("KH_DEBUG", default=False, cast=bool):
                doc.metadata["score__llm"] = results[idx]
            if include_doc:
                filtered_docs.append(doc)

        # prevent returning empty result
        if len(filtered_docs) == 0:
            filtered_docs = documents[: self.top_k]

        return filtered_docs
