from transformers import AutoTokenizer, LlamaForCausalLM
from huggingface_hub import login
import argparse as ap
from tqdm import tqdm
import itertools
import my_util
import torch
import json
import os

# Query2Doc using LLaMa 3.1 8B Instruct:
# - Given a query, generate an answer (question answering)
# - Do this for EVERY query
# - Should result in a more "symmetric" data...does this improve retrieval metrics?
def main():
	parser = ap.ArgumentParser('Query2Doc using Transformers')
	parser.add_argument('model_name_or_path', type=str, help='Model name or path')
	parser.add_argument('topics', type=str, help='Answers.json file to generate queries from.', nargs='+')
	parser.add_argument('-ext', type=str, help='Name extension to add to generated documents.')
	parser.add_argument('-t', '--token', type=str, help='HF token')
	parser.add_argument('-c', '--cache', type=str, help='HF_HOME/cache path.', default='.')
	parser.add_argument('-tc', '--clamp', type=int, help='Clamp query amount', default=None)
	args = parser.parse_args()

	# Set cache
	os.environ['HF_HOME'] = args.cache

	# Login
	if args.token:
		login(args.token)
	else:
		login()

	for index, topic_path in enumerate(args.topics):

		# Read topics for answer generation
		topics_dict = my_util.read_topics(args.topics)
		if args.clamp is not None:
			topics_dict = dict(itertools.islice(topics_dict.items(), args.clamp))

		ext = '_' + args.ext if args.ext else ''
		output_filename = f'GeneratedAnswers{ext}_{index + 1}'

		# As function says.
		llama_query2doc(args.model_name_or_path, topics_dict, output_filename)

# Generate queries to documents (answers)
def llama_query2doc(model_name_or_path, topics_dict, output_filename):
	# Get device and clear cache
	device = "cuda" if torch.cuda.is_available() else "cpu"
	torch.cuda.empty_cache()

	# Get tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map=device)
	tokenizer.pad_token = tokenizer.eos_token

	# Get model
	model = get_llama_model(model_name_or_path, device)

	# Our few shot messages "prompt." The text is reworded by ChatGPT.
	few_shot_messages = [
	{"role": "system",
	 "content": "You are a question generator assistant for puzzle and riddle answers. When given an answer you will generate a corresponding question. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a question."},
	# 66712
	{'role': 'user',
	 'content': "Film plots explained badly. Everybody slept for most of the movie. An Apple computer discovered some plant and eventually make everyone do exercise. It’s a Shakespeare adaptation but with cats and monkeys. 90+ year old government employee convinced a group of friends to betray his country. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Man killed a dog then died because a woman brought him home. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one."},
	# 66714
	{'role': 'assistant',
	 'content': "I'll have to do a little extra work in collating all the correct answers to earn my green tick, so here they are:  Everybody slept for most of the movie. Inception, where most of the movie happened in dreams, dreamt by the characters who were asleep. An Apple computer discovered some plant and eventually make everyone do exercise. WALL-E - humanity is obese, the earth's environment is shot, the titular robot discovers a plant, and at its conclusion they start restoring the environment and get people off their behinds! Courtesy of Phylyp. It’s a Shakespeare adaptation but with cats and monkeys. The Lion King. It's basically Hamlet with lions. Courtesy of Jaap Scherphuis. 90+ year old government employee convinced a group of friends to betray his country. Captain America: Civil War. Captain America, the world's oldest soldier, led the faction of superheroes who were against the Sokovia Accord as mandated by the UN. Sport athlete got caught by the police and forced to do manual labor, ended up learning humility and got sweet sponsorships. Cars. Lightning McQueen is a racer car, gets caught in Radiator Springs and has to repave the road, learns humility in the process. Gets sponsored by Rust-eeze. Courtesy of Thorbjorn Ravn Andersen. Man killed a dog then died because a woman brought him home. I am legend. Will Smith's character has to kill his infected dog, goes mad and is found by a woman. This woman brings him home, without covering their tracks, so they are tracked. Courtesy of Kamil Jurek. Genius with superiority complex abandoned job offers to chase some girl he met in a bar. Good Will Hunting (1997). Will (the protagonist) is a math prodigy with a superiority complex who rejects a lucrative job offer from the NSA and instead drives to California to reconnect with Skylar (a girl he met at a bar). Courtesy of ConjuringFrictionForces. Middle age man with unorthodox diet help chasing someone obsessed with skincare products. Silence of the Lambs: Hannibal Lecter most definitely has an unusual diet, and he helped Clarice Starling arrest the serial killer Buffalo Bill, who was seen putting on a lot of make-up. A son understood his immigrant father more after he went to Europe and fell in love with a girl. Godfather. Michael Corleone goes back to Italy and falls in love with an italian girl. Courtesy of Thorbjorn Ravn Andersen again. Two men arrived in Los Angeles seeking a young kid. You’d want to root for the more muscular one. Terminator 2: Judgment Day. The good robot (titular terminator) as played by the massive Arnold Schwarzenegger, is looking to save said young kid."}
	]

	generated_answers_dict = []
	for topic_id, topic in tqdm(list(topics_dict.items()), desc='Generating LLaMa Answer From Query', colour='blue'):
		# Tokenize and apply template to prompt
		dict_list_prompt = few_shot_messages + [{'role': 'user', 'content': str(topic)}]
		tokenized_prompt = tokenizer.apply_chat_template(dict_list_prompt,
														   add_generation_prompt=True,
														   return_tensors='pt',
														   padding=True
														   ).to(device=device)
		# Generate answer
		outputs = model.generate(inputs=tokenized_prompt,
								 generation_config=model.generation_config,
								 pad_token_id=tokenizer.eos_token_id)
		# Decode and append answer
		generated_answer = tokenizer.batch_decode(outputs[:, tokenized_prompt.shape[1]:],skip_special_tokens = True)[0]
		generated_answers_dict.append({'Id': topic_id, 'Text': generated_answer})

	# Dump to file
	with open(f'{output_filename}.json', 'w', encoding='utf-8') as outfile:
		json.dump(generated_answers_dict, outfile, indent = 4)
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
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()