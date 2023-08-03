import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_trainer import SpERTTrainer

import sys
sys.argv=['']
del sys

def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict_custom, arg_parser=arg_parser)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)
    
def __predict_custom(run_args):
    trainer = SpERTTrainer(run_args)

    text = 'The above mentioned patient attended our clinic today. On slit-lamp examination there has been observed a pterygium on the left eye. His vision is 6/5 right (unaided) and 6/9 left (unaided). Intraocular pressure was within normal range bilaterally. '
    trainer.predict_custom(dataset_path=text, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReaderCustom)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    
    #arg_parser.add_argument('mode', type=str, default='train', help="Mode: 'train' or 'eval'")
    #args, _ = arg_parser.parse_known_args()
    arg_parser.add_argument('-f')

    arg_parser.add_argument('--mode', type=str, default='predict', help="Mode: 'train' or 'eval'")
    arg_parser.add_argument('--config', type=str, default='configs/vabert_eval_biobert.conf', help="Config file path")
    #args = arg_parser.parse_args()
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")
