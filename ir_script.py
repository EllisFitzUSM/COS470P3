from bm25_ir import BM25
from tqdm import tqdm
import argparse as ap
import my_util
import json
import os

def main():
	parser = ap.ArgumentParser('Simple BM25 IR Script for Project 3.')
	parser.add_argument('corpus_path', help='Corpus File (representing document collection).')
	parser.add_argument('queries_path', help='Queries File.', nargs='+')
	parser.add_argument('-pti', '--pt_index', help='Path to PyTerrier index.', default=r'./pt_index')
	parser.add_argument('-name', help='Custom name to be appended to run names.', default='')
	parser.add_argument('-res', '--results_dir', help='Path to directory to save results.', default=r'.\results')
	parser.add_argument('-qt', '--query_tuple_corpus', action='store_true', help='Flag on if the corpus is actually "triple queries"')
	parser.add_argument('-tq', '--queries_are_topics', action='store_true', help='Flag on if the queries are actually "topic queries"')
	args = parser.parse_args()
	os.makedirs(args.results_dir, exist_ok=True)

	if args.query_tuple_corpus:
		corpus_dict_list = []
		generated_query_tuple_list = json.load(open(args.corpus_path, 'r', encoding='utf-8'))
		for query_tuple_dict in generated_query_tuple_list:
			corpus_dict_list.append({'Id': query_tuple_dict['Id'], 'Text': ' '.join([entry['content'] for entry in query_tuple_dict['Text']])})
		bm25 = BM25(corpus_dict_list, args.pt_index)
	else:
		bm25 = BM25(args.corpus_path, args.pt_index)

	for index, query_path in enumerate(tqdm(args.queries_path, desc='Reading Query Path')):
		if args.queries_are_topics:
			query_dict = my_util.read_topics(query_path)
		else:
			query_dict = get_query_dict(query_path)
		bm25_rankings = bm25.rank(query_dict)
		my_util.save_run(bm25_rankings, f'res_BM25_{args.name}_{index + 1}', args.results_dir)

# Can't quite use the util function I created because I saved the generated queries/answers differently...
def get_query_dict(query_path):
	query_list = json.load(open(query_path, 'r', encoding='utf-8'))
	query_dict = {}
	for query_pair in query_list:
		qid = query_pair['Id']
		text = query_pair['Text']
		query_dict[qid] = my_util.preprocess_text(text)
	return query_dict

if __name__ == '__main__':
	main()