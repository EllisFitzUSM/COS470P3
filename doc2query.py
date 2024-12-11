from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaForCausalLM, AutoTokenizer
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
	# msmarco_dict = []
	# beir_dict = []
	llama_list = []
	# Do all query-generation at the same time.
	try:
		# msmarco_process = mp.Process(target=msmarco_doc2query, args=(answers_dict,msmarco_dict,device))
		# beir_process = mp.Process(target=beir_doc2query, args=(answers_dict,beir_dict,device))
		model_kwargs = {'min_new_tokens': 128, 'max_new_tokens': 256, 'num_return_sequences': 1}
		llama_process = mp.Process(target=llama_doc2query, args=(llama_model_path,answers_dict,llama_list,'Doc2Queries.json',model_kwargs))

		# msmarco_process.start()
		# beir_process.start()
		llama_process.start()

		# msmarco_process.join()
		# beir_process.join()
		llama_process.join()
	except KeyboardInterrupt:
		sys.exit()

# MS-MARCO DocTTTTTQuery LM model
# def msmarco_doc2query(answers_dict, msmarco_dict, device):
# 	model_name = 'doc2query/msmarco-t5-base-v1'
# 	tokenizer = T5Tokenizer.from_pretrained(model_name)
# 	model = T5ForConditionalGeneration.from_pretrained(model_name)
# 	model.to(device)
# 	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating MS-MARCO Query From Doc', colour='blue'):
# 		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
# 		tokenized_answer = tokenized_answer.to('cuda:0')
# 		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
# 		msmarco_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
# 	with open('MSMARCO_Queries.json', 'w', encoding='utf-8') as outfile:
# 		json.dump(msmarco_dict, outfile, indent=4)
#
# # BeIR DocTTTTTQuery LM model (trained from above)
# def beir_doc2query(answers_dict, beir_dict, device):
# 	model_name = 'BeIR/query-gen-msmarco-t5-large-v1'
# 	tokenizer = T5Tokenizer.from_pretrained(model_name)
# 	model = T5ForConditionalGeneration.from_pretrained(model_name)
# 	model.to(device)
# 	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating BeIR Query From Doc', colour='blue'):
# 		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
# 		tokenized_answer = tokenized_answer.to('cuda:0')
# 		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
# 		beir_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
# 	with open('BeIR_Queries.json', 'w', encoding='utf-8') as outfile:
# 		json.dump(beir_dict, outfile, indent = 4)

# LLaMa 3.1 8B Instruct
# def llama_doc2query(llama_model_path, answers_dict, llama_dict, device):
# 	# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 	pipeline = transformers.pipeline(
# 		"text-generation",
# 		model=llama_model_path,
# 		model_kwargs={"torch_dtype": torch.bfloat16},
# 		device_map=device
# 	)
# 	pipeline.model.generation_config.pad_token_id = pipeline.model.generation_config.eos_token_id
# 	messages = [
# 		{"role": "system", "content": "You are a puzzle and riddle generator. When given a solved puzzle or riddle, you will generate a corresponding riddle or puzzle that the given answer solves. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a puzzle or riddle."},
# 		# 66714
# 		{'role': 'user', 'content': "Everybody slept for most of the movie. Inception, where most of the movie happened in dreams, dreamt by the characters who were asleep. An Apple computer discovered some plant and eventually make everyone do exercise. WALL-E - humanity is obese, the earth's environment is shot, the titular robot discovers a plant, and at its conclusion they start restoring the environment and get people off their behinds! Courtesy of Phylyp. It’s a Shakespeare adaptation but with cats and monkeys. The Lion King. It's basically Hamlet with lions. Courtesy of Jaap Scherphuis. 90+ year old government employee convinced a group of friends to betray his country. Captain America: Civil War. Captain America, the world's oldest soldier, led the faction of superheroes who were against the Sokovia Accord as mandated by the UN. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Cars. Lightning McQueen is a racer car, gets caught in Radiator Springs and has to repave the road, learns humility in the process. Gets sponsored by Rust-eeze. Courtesy of Thorbjorn Ravn Andersen. Man killed a dog then died because a woman brought him home. I am legend. Will Smith's character has to kill his infected dog, goes mad and is found by a woman. This woman brings him home, without covering their tracks, so they are tracked. Courtesy of Kamil Jurek. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Good Will Hunting (1997). Will (the protagonist) is a math prodigy with a superiority complex who rejects a lucrative job offer from the NSA and instead drives to California to reconnect with Skylar (a girl he met at a bar). Courtesy of ConjuringFrictionForces. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. Silence of the Lambs: Hannibal Lecter most definitely has an unusual diet, and he helped Clarice Starling arrest the serial killer Buffalo Bill, who was seen putting on a lot of make-up. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Godfather. Michael Corleone goes back to Italy and falls in love with an italian girl. Courtesy of Thorbjorn Ravn Andersen again. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one. Terminator 2: Judgment Day. The good robot (titular terminator) as played by the massive Arnold Schwarzenegger, is looking to save said young kid."},
# 		# 66712
# 		{'role': 'assistant', 'content': "Film plots explained badly. Everybody slept for most of the movie. An Apple computer discovered some plant and eventually make everyone do exercise. It’s a Shakespeare adaptation but with cats and monkeys. 90+ year old government employee convinced a group of friends to betray his country. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Man killed a dog then died because a woman brought him home. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one."}
# 	]
# 	for answer_id, answer in tqdm(list(answers_dict.items()), desc='Generating LLaMa Query From Doc', colour='blue'):
# 		outputs = pipeline(messages + [{'role': 'user', 'content': answer}], max_new_tokens=256, num_return_sequences=3)
# 		llama_dict.append({'Id': answer_id, 'Text': [output['generated_text'][-1] for output in outputs]})
# 	with open('LLaMa_Queries.json', 'w', encoding='utf-8') as outfile:
# 		json.dump(llama_dict, outfile, indent = 4)


# Generate queries to documents (answers)
def llama_doc2query(model_name_or_path, answers_dict, llama_list, output_filename, kwargs):
	# Get device and clear cache
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# torch.cuda.empty_cache()

	# Get tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map=device)
	tokenizer.pad_token = tokenizer.eos_token

	# Get model
	model = get_llama_model(model_name_or_path, device, **kwargs)

	# Our few shot messages "prompt." The text is reworded by ChatGPT.
	few_shot_messages = [
		{"role": "system",
		 "content": "You are a puzzle and riddle generator. When given a solved puzzle or riddle, you will generate a corresponding riddle or puzzle that the given answer solves. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a puzzle or riddle."},
		# 66714
		{'role': 'user',
		 'content': "Everybody slept for most of the movie. Inception, where most of the movie happened in dreams, dreamt by the characters who were asleep. An Apple computer discovered some plant and eventually make everyone do exercise. WALL-E - humanity is obese, the earth's environment is shot, the titular robot discovers a plant, and at its conclusion they start restoring the environment and get people off their behinds! Courtesy of Phylyp. It’s a Shakespeare adaptation but with cats and monkeys. The Lion King. It's basically Hamlet with lions. Courtesy of Jaap Scherphuis. 90+ year old government employee convinced a group of friends to betray his country. Captain America: Civil War. Captain America, the world's oldest soldier, led the faction of superheroes who were against the Sokovia Accord as mandated by the UN. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Cars. Lightning McQueen is a racer car, gets caught in Radiator Springs and has to repave the road, learns humility in the process. Gets sponsored by Rust-eeze. Courtesy of Thorbjorn Ravn Andersen. Man killed a dog then died because a woman brought him home. I am legend. Will Smith's character has to kill his infected dog, goes mad and is found by a woman. This woman brings him home, without covering their tracks, so they are tracked. Courtesy of Kamil Jurek. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Good Will Hunting (1997). Will (the protagonist) is a math prodigy with a superiority complex who rejects a lucrative job offer from the NSA and instead drives to California to reconnect with Skylar (a girl he met at a bar). Courtesy of ConjuringFrictionForces. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. Silence of the Lambs: Hannibal Lecter most definitely has an unusual diet, and he helped Clarice Starling arrest the serial killer Buffalo Bill, who was seen putting on a lot of make-up. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Godfather. Michael Corleone goes back to Italy and falls in love with an italian girl. Courtesy of Thorbjorn Ravn Andersen again. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one. Terminator 2: Judgment Day. The good robot (titular terminator) as played by the massive Arnold Schwarzenegger, is looking to save said young kid."},
		# 66712
		{'role': 'assistant',
		 'content': "Film plots explained badly. Everybody slept for most of the movie. An Apple computer discovered some plant and eventually make everyone do exercise. It’s a Shakespeare adaptation but with cats and monkeys. 90+ year old government employee convinced a group of friends to betray his country. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Man killed a dog then died because a woman brought him home. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one."}
	]

	for topic_id, topic in tqdm(list(answers_dict.items()), desc='Generating LLaMa Answer From Query', colour='blue'):
		# Tokenize and apply template to prompt
		dict_list_prompt = few_shot_messages + [{'role': 'user', 'content': str(topic)}]
		tokenized_prompt = tokenizer.apply_chat_template(dict_list_prompt,
														   add_generation_prompt=True,
														   return_tensors='pt',
														   padding=True
														   ).to(device=device)
		# Generate questions
		outputs = model.generate(inputs=tokenized_prompt,
								 generation_config=model.generation_config,
								 pad_token_id=tokenizer.eos_token_id)
		# Decode and append answer
		generated_answers = tokenizer.batch_decode(outputs[:, tokenized_prompt.shape[1]:],skip_special_tokens = True)
		llama_list.append({'Id': topic_id, 'Text': generated_answers})

	# Dump to file
	with open(f'{output_filename}.json', 'w', encoding='utf-8') as outfile:
		json.dump(llama_list, outfile, indent = 4)
	outfile.close()

# Get the LLaMa file and apply a few kwargs.
def get_llama_model(model_name_or_path, device, **kwargs):
	model = LlamaForCausalLM.from_pretrained(model_name_or_path,
											 device_map=device,
											 torch_dtype=kwargs.get('torch_dtype', torch.bfloat16))
	model.generation_config.min_new_tokens = kwargs.get('min_new_tokens', 512)
	model.generation_config.max_new_tokens = kwargs.get('max_new_tokens', 1024)
	model.generation_config.do_sample = kwargs.get('do_sample', True)
	model.generation_config.num_return_sequences = kwargs.get('num_return_sequences', 1)
	model = model.to(device)
	return model

if __name__ == '__main__':
	# torch.multiprocessing.set_start_method('spawn', force=True)
	main()