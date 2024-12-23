# COS470P3: Stack Exchange LLM Information Retrieval

| x | Model | NDCG@5 | NDCG@10 | P@5 | P@10 | MAP | BPref | MRR |
|:- | :----  | :----- | :------ | :-- | :--- | :-- | :---- | :-- |
| a | results\res_BM25_1   | 0.408ᵇᶜᵈ | 0.428ᵇᶜᵈ | 0.372ᵇᶜᵈ | 0.246ᵇᶜᵈ | 0.341ᵇᶜᵈ | nan | 0.708ᵇᶜᵈ |
| b | results\res_BM25_doc2query_LLaMa_best_1 | 0.239 | 0.261 | 0.225 | 0.159 | 0.196 | nan | 0.485 |
| c |results\res_BM25_doc2query_LLaMa_triplet_1 | 0.311ᵇ | 0.334ᵇ | 0.297ᵇ | 0.204ᵇ | 0.268ᵇ | nan | 0.581ᵇ |
| d | results\res_BM25_query2doc_1 | 0.337ᵇᶜ | 0.359ᵇᶜ | 0.307ᵇ | 0.211ᵇ | 0.283ᵇ | nan | 0.607ᵇ |

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