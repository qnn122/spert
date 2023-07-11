"""Make data for joint training of NER and RE from Label Studio annotation json files

Example:
	python scripts/make_data_from_labelstudio_joint_training.py \
		--data_dir /home/quangng/vabert/data/v1.2 \
		--output_dir data/datasets/vabert \
		--tokenizer bert-base-cased

"""
import srsly
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from tqdm import tqdm
from itertools import permutations
import random
import argparse
import csv
from transformers import AutoTokenizer


_VABERT_TYPES = {
	"entities": {
		"VA": {"short": "VA", "verbose": "Visual Acuity"},
		"LAT": {"short": "LAT", "verbose": "Laterality (Eye side)"},
		"ET": {"short": "ET", "verbose": "Exam Type"},
	},
	"relations": {
		"va-lat": {"short": "va-lat", "verbose": "Visual Acuity - Laterality", "symmetric": False},
		"va-et": {"short": "va-et", "verbose": "Visual Acuity - Exam Type", "symmetric": False},
	}
}

	
def get_annotation(sample):
	"""Load text, entities and relations from Label Studio anntoation json files
	"""
	text = sample['data']['text']
	anns = sample['annotations'][0]['result']
	_id = sample['id']

	entities = []
	relations = []
	for a in anns:
		# Entities
		if a['type'] == 'labels':
			ent = a['value']
			if 'labels' in ent:
				ent['type'] = ent['labels'][0]
				del ent['labels']
			else:
				ent['type'] = ent['type']
			ent['id'] = a['id']
			entities.append(ent)

		# Relations
		if a['type'] == 'relation':
			rel = a
			rel['type'] = rel['labels'][0]	
			del rel['labels']
			if rel['type'] == 'no-rel':
				continue
			relations.append(rel)

	return text, entities, relations, _id

def generate_neg_samples(rels, ents, num_samples='balanced'):
	ent_ids = [e['id'] for e in ents]
	permus = list(permutations(ent_ids, 2))
	pos_samples = [(rel['from_id'], rel['to_id']) for rel in rels]
	negs_ = list(set(permus).difference(set(pos_samples)))

	if num_samples == 'balanced':
		negs = random.choices(negs_, k=len(pos_samples))	
	elif num_samples == 'all':
		negs = negs_

	#neg_samples = []
	neg_samples = [{"from_id":sam[0],
					"to_id": sam[1],
					"type": "relation",
					"labels":["no-rel"]} for sam in negs] # omit "direction"

	return neg_samples

def split_and_save(df, data_dir, task_name, classes, test_size=0.3):
	save_dir = os.path.join(data_dir, task_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	ext = '.csv'
	sep = ','
	header = True
	if task_name == 're': # relation extraction is slightly different
		ext = '.tsv'
		sep = '\t'
		header = False
		classes.sort()
		#classes.insert(0, 'other')

	df.to_csv(os.path.join(save_dir, 'all' + ext), sep=sep, index=False, header=header)
	train, test = train_test_split(df, test_size=test_size)

	train.to_csv(os.path.join(save_dir, 'train' + ext), sep=sep, index=False, header=header)
	test.to_csv(os.path.join(save_dir, 'test' + ext), sep=sep, index=False, header=header)

	with open(os.path.join(save_dir, 'label.txt'), 'w') as f:
		for item in classes:
			f.write("%s\n" % item)

def char2tok(char_data, tokenizer):
	"""Convert character-level NER annotation to token-level one.

	Retunrs:
		tokens: list of tokens processed by the given `tokenizer`
		entites: list of token-level entities with the following format:
			{
				"type": <entity type>,
				"start": <start token>,
				"end": <end token>
			}

	Example:
		>>> char_data = {
				"text": "John Wilkes Bootes, who assassinated President Lincoln, was an actor.",
				"entities": [{"type": "Peop", "start": 0, "end": 18}, {"type": "Peop", "start": 37, "end": 54}]
			}
		>>> char2tok(char_data, tokenizer)
		(['John', 'Wilkes', 'Bo', '##otes', ',', 'who', 'assassinated', 'President', 'Lincoln', ',', 'was', 'an', 'actor', '.'],
		[{'type': 'Peop', 'start': 0, 'end': 4}, {'type': 'Peop', 'start': 7, 'end': 9}])
	"""
	# Tokenize the text
	tokens = tokenizer.tokenize(char_data["text"])
	tok_data = tokenizer(char_data["text"])

	# Convert the character-level entities to token-level
	tok_entities = []
	for ent in char_data["entities"]:
		tok_start = tok_data.char_to_token(ent["start"]) - 1 # retunred index starts from 1, need to minus 1
		tok_end = tok_data.char_to_token(ent["end"]-1)
		tok_entities.append({"type": ent["type"], "start": tok_start, "end": tok_end})

	# Return the tokenized data
	return tokens, tok_entities

def make_data(args):
	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

	# Load data
	data_dir = args.data_dir
	json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]

	# load all lines in all json files using srsly.read_json
	lines = []
	for file in json_files:
		lines.extend(srsly.read_json(os.path.join(data_dir, file)))

	# 
	data_all = []
	for line in tqdm(lines):
		text, ents, rels, _id = get_annotation(line)
		if len(ents) == 0:
			continue
		data = {"original_id": _id}

		# Preprocessing text ---------------------------------------------------
		text = text.replace('#', '*') 
		text = text.replace('\t', ' ') 
		text = text.replace('\n', ' ') 

		# Make NER data --------------------------------------------------------
		tokens, tok_entities = char2tok({"text": text, "entities": ents}, tokenizer)
		data["tokens"] = tokens
		data["entities"] = tok_entities
		
		# Make RE data ---------------------------------------------------------
		ents_ids = [e['id'] for e in ents]
		relations = [{'type': rel['type'],
						'head': ents_ids.index(rel['to_id']),
						'tail': ents_ids.index(rel['from_id'])}
						for rel in rels
		]
		data["relations"] = relations
		data_all.append(data)

	# Split data ---------------------------------------------------------------
	train, test = train_test_split(data_all, test_size=0.2)
	train, dev = train_test_split(train, test_size=0.1)

	# Save output -------------------------------------------------------------
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	srsly.write_json(os.path.join(output_dir, 'train.json'), train)
	srsly.write_json(os.path.join(output_dir, 'test.json'), test)
	srsly.write_json(os.path.join(output_dir, 'dev.json'), dev)
	srsly.write_json(os.path.join(output_dir, 'all.json'), data_all)
	srsly.write_json(os.path.join(output_dir, 'types.json'), _VABERT_TYPES)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default='/home/quangng/vabert/data/v1.2', type=str, help="Data directory")
	parser.add_argument("--output_dir", default='data/datasets/vabert', type=str, help="Output directory")
	parser.add_argument("--tokenizer_name", default='bert-base-cased', type=str, help="Tokenizer name or path")

	args = parser.parse_args()

	make_data(args)