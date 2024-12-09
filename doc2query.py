from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import login
import multiprocessing as mp
import argparse as ap
from tqdm import tqdm
import transformers
import itertools
import my_util
import torch
import json
import sys
import os

# This was more experiment-y.
# Messed around with multiprocessing,
# wanted to see if I could really maximize the best scores given no time limits for A5.
def main():
	parser = ap.ArgumentParser('Doc2Query using Transformers')
	parser.add_argument('answers', type=str, help='Answers.json file to generate queries from.')
	parser.add_argument('-t', '--token', type=str, help='HF token')
	parser.add_argument('-c', '--cache', type=str, help='HF_HOME/cache path.', default='.')
	parser.add_argument('-tc', '--clamp', type=int, help='Clamp answer amount', default=None)
	args = parser.parse_args()

	# Set cache
	os.environ['HF_HOME'] = args.cache

	# Login
	if args.token:
		login(args.token)
	else:
		login()

	# Read answers for query generation
	answers_dict = my_util.read_answers(args.answers)
	if args.clamp is not None:
		answers_dict = dict(itertools.islice(answers_dict.items(), args.clamp))

	# As functions says
	doc2query(answers_dict)

# Generate queries from documents (answers).
def doc2query(answers_dict):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	msmarco_dict = []
	beir_dict = []
	llama_dict = []
	# Do all query-generation at the same time.
	try:
		msmarco_process = mp.Process(target=msmarco_doc2query, args=(answers_dict,msmarco_dict,device))
		beir_process = mp.Process(target=beir_doc2query, args=(answers_dict,beir_dict,device))
		llama_process = mp.Process(target=llama_doc2query, args=(answers_dict,llama_dict,device))

		msmarco_process.start()
		beir_process.start()
		llama_process.start()

		msmarco_process.join()
		beir_process.join()
		llama_process.join()
	except KeyboardInterrupt:
		sys.exit()

# MS-MARCO DocTTTTTQuery LM model
def msmarco_doc2query(answers_dict, msmarco_dict, device):
	model_name = 'doc2query/msmarco-t5-base-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to(device)
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating MS-MARCO Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		msmarco_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('MSMARCO_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(msmarco_dict, outfile, indent=4)

# BeIR DocTTTTTQuery LM model (trained from above)
def beir_doc2query(answers_dict, beir_dict, device):
	model_name = 'BeIR/query-gen-msmarco-t5-large-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to(device)
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating BeIR Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		beir_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('BeIR_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(beir_dict, outfile, indent = 4)

# LLaMa 3.1 8B Instruct
def llama_doc2query(answers_dict, llama_dict, device):
	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	pipeline = transformers.pipeline(
		"text-generation",
		model=model_name,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map=device
	)
	pipeline.model.generation_config.pad_token_id = pipeline.model.generation_config.eos_token_id
	# These examples used for few-show was a regenerated query and answer pair from ChatGPT
	messages = [
		{"role": "system", "content": "You are question generator assistant for travelling answers. When given an answer you will generate a corresponding question. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a question."},
		{'role': 'user', 'content': "Practices regarding complimentary tap water in Europe vary widely, with no universal custom. While free water isn’t exclusive to Finland or Scandinavia, laws and traditions differ by country. some places, serving tap water is required by law, such as the UK (for premises serving alcohol), France (where pitchers are often provided automatically with meals), Hungary, and Spain. In Finland, Norway, Sweden, Denmark, and Slovenia, free water is very common. In countries like Switzerland, free tap water is offered inconsistently, while in the Netherlands, Germany, Luxembourg, Italy, and Belgium, it’s less common, and patrons typically order paid drinks. Some restaurants in these regions may refuse or appear surprised if asked for free water. Even in countries where laws mandate free tap water, exceptions occur, such as in mountain lodges or upscale venues. High-end restaurants may expect customers to purchase drinks, sometimes offering filtered or carbonated water as a paid alternative. Lastly, in places like Austria, France, and Italy, serving a glass of water alongside coffee is customary and generally well-accepted."},
		{'role': 'assistant', 'content': "How frequently do restaurants in Europe provide complimentary drinking water upon request? When I visited Helsinki, I noticed restaurants often provided free water with orders. This included places like McDonald’s, where my friend requested tap water, and it was served without charge. Some restaurants even encouraged this practice, offering water refill stations with clean glasses or placing glass jugs of water near the soft drink area for self-service. I haven’t observed this elsewhere in Europe, though my travels are limited. Is free water for customers a common practice across Europe, or is it specific to Finland or Scandinavia?"}
	]
	for answer_id, answer in tqdm(list(answers_dict.items()), desc='Generating LLaMa Query From Doc', colour='blue'):
		outputs = pipeline(messages + [{'role': 'user', 'content': answer}], max_new_tokens=256, num_return_sequences=3)
		llama_dict.append({'Id': answer_id, 'Text': [output['generated_text'][-1] for output in outputs]})
	with open('collections/LLaMa_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(llama_dict, outfile, indent = 4)

if __name__ == '__main__':
	main()