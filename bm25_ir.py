import pyterrier as pt
import pandas as pd
import my_util
import json
import os

# Simple PyTerrier BM25-Okapi Wrapper Class
class BM25(object):

	def __init__(self, corpus_path_or_list, index_save):
		self.answers_path_or_list = corpus_path_or_list
		self.index_ref = self.get_bm25_index_ref(index_save)

	def get_bm25_index_ref(self, index_save):
		index_abs_path = os.path.abspath(index_save)
		if not os.path.exists(os.path.join(index_abs_path, 'data.properties')): # Feels unsafe/not backwards compatible.
			os.makedirs(index_abs_path, exist_ok=True)
			pt_indexer = pt.IterDictIndexer(index_abs_path,
											verbose=True,
											overwrite=True,
											stopwords=my_util.get_stopwords(),
											tokeniser='english')
			if isinstance(self.answers_path_or_list, list):
				docs_df: pd.DataFrame = pd.DataFrame(self.answers_path_or_list)
			elif isinstance(self.answers_path_or_list, str):
				docs_df: pd.DataFrame = pd.DataFrame(json.load(open(self.answers_path_or_list, 'r', encoding='utf-8')))
			else:
				raise Exception('The answers_path_or_list has to be a list or a string.')
			docs_df.rename({'Id': 'docno', 'Text': 'text'}, axis='columns', inplace=True)
			pt_indexer.index(docs_df[['text', 'docno']].to_dict(orient='records'))
		return os.path.join(index_abs_path, 'data.properties')

	def rank(self, topics_dict):
		retriever = pt.terrier.Retriever(self.index_ref, num_results=100, wmodel='BM25')
		topics_df = pd.DataFrame(topics_dict.items(), columns=['qid', 'query'])
		rankings = retriever.transform(topics_df)
		rankings_dict = rankings.groupby('qid').apply(lambda group: dict(zip(group['docno'], group['score']))).to_dict()
		return rankings_dict