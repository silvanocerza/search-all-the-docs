---
title: SEARCH ALL THE DOCS
emoji: 🔎
colorFrom: yellow
colorTo: pink
python_version: 3.11
sdk: streamlit
sdk_version: 1.27.2
app_file: main.py
pinned: false
---

![SEARCH ALL THE DOCS](meme.jpg)

## Getting started

First create your virtual env so you don't pollute your OS environment.
This demo has only been tested with Python 3.11, so I suggest you use that.

```shell
mkvirtualenv search-all-the-docs
workon search-all-the-docs
```

Install the dependencies:

```shell
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI key:

```
OPENAI_API_KEY="<your_key_here>"
```

And you're good to go!

```shell
streamlit run main.py
```
