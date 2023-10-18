from typing import List, Tuple
from pathlib import Path
import subprocess
import os

from dotenv import load_dotenv
from haystack.preview import Pipeline
from haystack.preview.dataclasses import GeneratedAnswer
from haystack.preview.components.retrievers import MemoryBM25Retriever
from haystack.preview.components.generators.openai.gpt import GPTGenerator
from haystack.preview.components.builders.answer_builder import AnswerBuilder
from haystack.preview.components.builders.prompt_builder import PromptBuilder
from haystack.preview.components.preprocessors import (
    DocumentCleaner,
    TextDocumentSplitter,
)
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.components.file_converters import TextFileToDocument
from haystack.preview.document_stores.memory import MemoryDocumentStore
import streamlit as st

# Load the environment variables, we're going to need it for OpenAI
load_dotenv()

# This is the list of documentation that we're going to fetch
DOCUMENTATIONS = [
    ("DocArray", "https://github.com/docarray/docarray", "./docs/**/*.md"),
    ("Streamlit", "https://github.com/streamlit/docs", "./content/**/*.md"),
    ("Jinja", "https://github.com/pallets/jinja", "./docs/**/*.rst"),
    ("Pandas", "https://github.com/pandas-dev/pandas", "./docs/source/**/*.rst"),
    (
        "Elasticsearch",
        "https://github.com/elastic/elasticsearch",
        "./docs/**/*.asciidoc",
    ),
    ("NumPy", "https://github.com/numpy/numpy", "./doc/**/*.rst"),
]


@st.cache_data(show_spinner=False)
def fetch(documentations: List[Tuple[str, str, str]]):
    paths = []
    for name, url, pattern in documentations:
        st.write(f"Fetching {name} repository")
        repo = Path(__file__).parent / "downloaded_docs" / name
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
        paths.extend(repo.glob(pattern))

    return paths


@st.cache_resource(show_spinner=False)
def document_store():
    # We're going to store the processed documents in here
    return MemoryDocumentStore()


@st.cache_resource(show_spinner=False)
def index_files(files):
    # We create some components
    text_converter = TextFileToDocument(progress_bar=False)
    document_cleaner = DocumentCleaner()
    document_splitter = TextDocumentSplitter()
    document_writer = DocumentWriter(
        document_store=document_store(), policy="overwrite"
    )

    # And our pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", text_converter)
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("converter", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")

    # And now we clone and save the documentation in our MemoryDocumentStore
    indexing_pipeline.run({"converter": {"paths": files}})


def search(question: str) -> GeneratedAnswer:
    retriever = MemoryBM25Retriever(document_store=document_store(), top_k=5)

    template = """Take a deep breath and think then answer given the context
    Context: {{ documents|map(attribute='text')|replace('\n', ' ')|join(';') }}
    Question: {{ query }}
    Answer:
    """
    prompt_builder = PromptBuilder(template)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    generator = GPTGenerator(api_key=OPENAI_API_KEY)
    answer_builder = AnswerBuilder()

    pipe = Pipeline()

    pipe.add_component("docs_retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("gpt35", generator)
    pipe.add_component("answer_builder", answer_builder)

    pipe.connect("docs_retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "gpt35.prompt")
    pipe.connect("docs_retriever.documents", "answer_builder.documents")
    pipe.connect("gpt35.replies", "answer_builder.replies")
    res = pipe.run(
        {
            "docs_retriever": {"query": question},
            "prompt_builder": {"query": question},
            "answer_builder": {"query": question},
        }
    )
    return res["answer_builder"]["answers"][0]


with st.status(
    "Downloading documentation files...",
    expanded=st.session_state.get("expanded", True),
) as status:
    files = fetch(DOCUMENTATIONS)
    status.update(label="Indexing documentation...")
    index_files(files)
    status.update(
        label="Download and indexing complete!", state="complete", expanded=False
    )
    st.session_state["expanded"] = False


st.header("Documentation finder", divider="rainbow")

st.caption(
    f"Use this to search answers for {', '.join([d[0] for d in DOCUMENTATIONS])}"
)

if question := st.text_input(
    label="What do you need to know?", placeholder="What is a DataFrame?"
):
    with st.spinner("Waiting"):
        answer = search(question)

    if not st.session_state.get("run_once", False):
        st.balloons()
        st.session_state["run_once"] = True

    print(answer.data)
    st.markdown(answer.data)
    with st.expander("See sources:"):
        for document in answer.documents:
            url_source = document.metadata.get("url_source", "")
            content = f"{url_source}: {document.text}" if url_source else document.text
            if document.metadata.get("type") == "md":
                st.markdown(content)
            else:
                st.write(content)
            st.divider()
