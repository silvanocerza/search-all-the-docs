import subprocess
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import GeneratedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

# Load the environment variables, we're going to need it for OpenAI
load_dotenv()

# This is the list of documentation that we're going to fetch
DOCUMENTATIONS = [
    (
        "DocArray",
        "https://github.com/docarray/docarray",
        "./docs/**/*.md",
    ),
    (
        "Streamlit",
        "https://github.com/streamlit/docs",
        "./content/**/*.md",
    ),
    (
        "Jinja",
        "https://github.com/pallets/jinja",
        "./docs/**/*.rst",
    ),
    (
        "Pandas",
        "https://github.com/pandas-dev/pandas",
        "./doc/source/**/*.rst",
    ),
    (
        "Elasticsearch",
        "https://github.com/elastic/elasticsearch",
        "./docs/**/*.asciidoc",
    ),
    (
        "NumPy",
        "https://github.com/numpy/numpy",
        "./doc/**/*.rst",
    ),
]

DOCS_PATH = Path(__file__).parent / "downloaded_docs"


@st.cache_data(show_spinner=False)
def fetch(documentations: List[Tuple[str, str, str]]):
    files = []
    # Create the docs path if it doesn't exist
    DOCS_PATH.mkdir(parents=True, exist_ok=True)

    for name, url, pattern in documentations:
        st.write(f"Fetching {name} repository")
        repo = DOCS_PATH / name
        # Attempt cloning only if it doesn't exist
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            cwd=repo,
        )
        branch = res.stdout.strip()
        for p in repo.glob(pattern):
            data = {
                "path": p,
                "meta": {
                    "url_source": f"{url}/tree/{branch}/{p.relative_to(repo)}",
                    "suffix": p.suffix,
                },
            }
            files.append(data)

    return files


@st.cache_resource(show_spinner=False)
def document_store(index: str = "documentation"):
    # We're going to store the processed documents in here
    return InMemoryDocumentStore(index=index)


@st.cache_resource(show_spinner=False)
def index_files(files):
    # We create some components
    text_converter = TextFileToDocument()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter()
    document_writer = DocumentWriter(
        document_store=document_store(), policy=DuplicatePolicy.OVERWRITE
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

    # And now we save the documentation in our InMemoryDocumentStore
    paths = []
    meta = []
    for f in files:
        paths.append(f["path"])
        meta.append(f["meta"])
    indexing_pipeline.run(
        {
            "converter": {
                "sources": paths,
                "meta": meta,
            }
        }
    )


def search(question: str) -> GeneratedAnswer:
    retriever = InMemoryBM25Retriever(document_store=document_store(), top_k=5)

    template = (
        "Using the information contained in the context, give a comprehensive answer to the question."
        "If the answer cannot be deduced from the context, do not give an answer."
        "Context: {{ documents|map(attribute='content')|replace('\n', ' ')|join(';') }}"
        "Question: {{ query }}"
        "Answer:"
    )
    prompt_builder = PromptBuilder(template)

    generator = OpenAIGenerator(model="gpt-4o")
    answer_builder = AnswerBuilder()

    query_pipeline = Pipeline()

    query_pipeline.add_component("docs_retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", generator)
    query_pipeline.add_component("answer_builder", answer_builder)

    query_pipeline.connect("docs_retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    query_pipeline.connect("docs_retriever.documents", "answer_builder.documents")
    query_pipeline.connect("llm.replies", "answer_builder.replies")
    res = query_pipeline.run({"query": question})
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


st.header("🔎 Documentation finder", divider="rainbow")

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

    st.markdown(answer.data)
    with st.expander("See sources:"):
        for document in answer.documents:
            url_source = document.meta.get("url_source", "")
            st.write(url_source)
            st.text(document.content)
            st.divider()
