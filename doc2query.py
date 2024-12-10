from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaForCausalLM
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
	parser.add_argument('llama_model_path', type=str, help='Path to local pretrained LLaMa model.')
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
	doc2query(args.llama_model_path, answers_dict)

# Generate queries from documents (answers).
def doc2query(llama_model_path, answers_dict):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# msmarco_dict = []
	# beir_dict = []
	llama_dict = []
	# Do all query-generation at the same time.
	try:
		# msmarco_process = mp.Process(target=msmarco_doc2query, args=(answers_dict,msmarco_dict,device))
		# beir_process = mp.Process(target=beir_doc2query, args=(answers_dict,beir_dict,device))
		llama_process = mp.Process(target=llama_doc2query, args=(llama_model_path,answers_dict,llama_dict,device))

		# msmarco_process.start()
		# beir_process.start()
		llama_process.start()

		# msmarco_process.join()
		# beir_process.join()
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
def llama_doc2query(llama_model_path, answers_dict, llama_dict, device):
	# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	pipeline = transformers.pipeline(
		"text-generation",
		model=LlamaForCausalLM.from_pretrained(llama_model_path, device_map=device),
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map=device
	)
	pipeline.model.generation_config.pad_token_id = pipeline.model.generation_config.eos_token_id
	messages = [
		{"role": "system", "content": "You are a question generator assistant for puzzle and riddle answers. When given an answer you will generate a corresponding question. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a question."},
		# 66714
		{'role': 'user', 'content': "I'll have to do a little extra work in collating all the correct answers to earn my green tick, so here they are:  Everybody slept for most of the movie. Inception, where most of the movie happened in dreams, dreamt by the characters who were asleep. An Apple computer discovered some plant and eventually make everyone do exercise. WALL-E - humanity is obese, the earth's environment is shot, the titular robot discovers a plant, and at its conclusion they start restoring the environment and get people off their behinds! Courtesy of Phylyp. It’s a Shakespeare adaptation but with cats and monkeys. The Lion King. It's basically Hamlet with lions. Courtesy of Jaap Scherphuis. 90+ year old government employee convinced a group of friends to betray his country. Captain America: Civil War. Captain America, the world's oldest soldier, led the faction of superheroes who were against the Sokovia Accord as mandated by the UN. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Cars. Lightning McQueen is a racer car, gets caught in Radiator Springs and has to repave the road, learns humility in the process. Gets sponsored by Rust-eeze. Courtesy of Thorbjorn Ravn Andersen. Man killed a dog then died because a woman brought him home. I am legend. Will Smith's character has to kill his infected dog, goes mad and is found by a woman. This woman brings him home, without covering their tracks, so they are tracked. Courtesy of Kamil Jurek. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Good Will Hunting (1997). Will (the protagonist) is a math prodigy with a superiority complex who rejects a lucrative job offer from the NSA and instead drives to California to reconnect with Skylar (a girl he met at a bar). Courtesy of ConjuringFrictionForces. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. Silence of the Lambs: Hannibal Lecter most definitely has an unusual diet, and he helped Clarice Starling arrest the serial killer Buffalo Bill, who was seen putting on a lot of make-up. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Godfather. Michael Corleone goes back to Italy and falls in love with an italian girl. Courtesy of Thorbjorn Ravn Andersen again. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one. Terminator 2: Judgment Day. The good robot (titular terminator) as played by the massive Arnold Schwarzenegger, is looking to save said young kid."},
		# 66712
		{'role': 'assistant', 'content': "Film plots explained badly. Everybody slept for most of the movie. An Apple computer discovered some plant and eventually make everyone do exercise. It’s a Shakespeare adaptation but with cats and monkeys. 90+ year old government employee convinced a group of friends to betray his country. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Man killed a dog then died because a woman brought him home. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one."}
	]
	for answer_id, answer in tqdm(list(answers_dict.items()), desc='Generating LLaMa Query From Doc', colour='blue'):
		outputs = pipeline(messages + [{'role': 'user', 'content': answer}], max_new_tokens=256, num_return_sequences=3)
		llama_dict.append({'Id': answer_id, 'Text': [output['generated_text'][-1] for output in outputs]})
	with open('collections/LLaMa_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(llama_dict, outfile, indent = 4)

if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()