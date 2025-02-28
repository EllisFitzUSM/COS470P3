# COS470P3: Stack Exchange LLM Information Retrieval

This repository demonstrates using an LLM to rewrite a large dataset to improve performance and metrics in an information retrieval task. The repository uses BM25-Okapi with the given question-answering dataset as the baseline. Two methods are used.
1. Doc2Query
- Uses Meta Llama to convert the answers dataset to questions, with the goal of increasing the semantic similarity to make the IR task symmetric (question to question) instead of asymmetric (question to answer).
2. Query2Doc
- Uses Meta Llama to convert the questions dataset to answers, with the goal of increasing the semantic similarity to make the IR task symmetric (answer to answer) instead of asymmetric (question to answer).

What this repository finds is Doc2Query is less performant, as it removes context from the passages and in some cases favors noise over the important information you normally would want to rank higher. Though, the baseline is still incredibly strong, showcasing how replacing the passages with the newly generated text instead of appending it (query/document expansion) is heavily prone to hallucination.

| x | Model | NDCG@5 | NDCG@10 | P@5 | P@10 | MAP | BPref | MRR |
|:- | :----  | :----- | :------ | :-- | :--- | :-- | :---- | :-- |
| a | Baseline   | 0.408ᵇᶜᵈ | 0.428ᵇᶜᵈ | 0.372ᵇᶜᵈ | 0.246ᵇᶜᵈ | 0.341ᵇᶜᵈ | nan | 0.708ᵇᶜᵈ |
| b | Doc2Query Best | 0.239 | 0.261 | 0.225 | 0.159 | 0.196 | nan | 0.485 |
| c | Doc2Query Triplet | 0.311ᵇ | 0.334ᵇ | 0.297ᵇ | 0.204ᵇ | 0.268ᵇ | nan | 0.581ᵇ |
| d | Query2Doc | 0.337ᵇᶜ | 0.359ᵇᶜ | 0.307ᵇ | 0.211ᵇ | 0.283ᵇ | nan | 0.607ᵇ |

To install the correct dependencies:

    > pip install -r requirements.txt

To save a proper snapshot of the same LLM model:

    >>> from transformers import LLamaForCasualLM
    >>> import os
    >>> os.environ['HF_HOME] = /path/to/directory
    >>> model = LlamaForCasualLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')



## Run

There are multiple files to run:

- doc2query.py
- query2doc.py
- ir_script.py

### doc2query

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `answers`    | `str` | Paths to Answers.json file to rewrite. |
| `llama_model_path`    | `str` | Model name or path. |
| `-t`    | `str` | Hugging Face API Token. |
| `-c`    | `str` | Path to Hugging Face cache (AKA HF_HOME). |
| `-tc`    | `flag` | If you would like to clamp the # of queries generated |

### query2doc

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name_or_path`    | `str` | Model name or path. |
| `topics`    | `str / List[str]` | Paths to Topics.json files to rewrite. |
| `-t`    | `str` | Hugging Face API Token. |
| `-c`    | `str` | Path to Hugging Face cache (AKA HF_HOME). |
| `-tc`    | `flag` | If you would like to clamp the # of queries generated |

### ir_script

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `corpus_path`    | `str` | Path to corpus document/file. |
| `queries_path`    | `str / List[str]` | Paths to queries files. |
| `-pti`    | `str` | Path to PyTerrier generated index (make sure this correlates to corpus!) |
| `-name`    | `str` | Custom name to be appended to run names. |
| `-res`    | `str` | Path to results directory. |
| `-qt`    | `flag` | Flag on if the corpus is actually "triplet_queries" from doc2query. |
| `-tc`    | `flag` | Flag on if the queries are in "topics_#.json" format. |

For the three (3) files above, you can always run the python.exe and the file with `-h` as an argumetn to get the descriptions for all the arguments available. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
