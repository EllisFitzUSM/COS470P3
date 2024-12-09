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
		{"role": "system", "content": "You are question answerer assistant for travelling questions. When given an question you will generate a corresponding anwswer. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate an answer."},
        {'role': 'user', 'content': "How frequently do restaurants in Europe provide complimentary drinking water upon request? When I visited Helsinki, I noticed restaurants often provided free water with orders. This included places like McDonald’s, where my friend requested tap water, and it was served without charge. Some restaurants even encouraged this practice, offering water refill stations with clean glasses or placing glass jugs of water near the soft drink area for self-service. I haven’t observed this elsewhere in Europe, though my travels are limited. Is free water for customers a common practice across Europe, or is it specific to Finland or Scandinavia?"},
        {'role': 'assistant', 'content': "Practices regarding complimentary tap water in Europe vary widely, with no universal custom. While free water isn’t exclusive to Finland or Scandinavia, laws and traditions differ by country. some places, serving tap water is required by law, such as the UK (for premises serving alcohol), France (where pitchers are often provided automatically with meals), Hungary, and Spain. In Finland, Norway, Sweden, Denmark, and Slovenia, free water is very common. In countries like Switzerland, free tap water is offered inconsistently, while in the Netherlands, Germany, Luxembourg, Italy, and Belgium, it’s less common, and patrons typically order paid drinks. Some restaurants in these regions may refuse or appear surprised if asked for free water. Even in countries where laws mandate free tap water, exceptions occur, such as in mountain lodges or upscale venues. High-end restaurants may expect customers to purchase drinks, sometimes offering filtered or carbonated water as a paid alternative. Lastly, in places like Austria, France, and Italy, serving a glass of water alongside coffee is customary and generally well-accepted."}
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
	main()