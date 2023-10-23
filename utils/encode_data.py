
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import string
import time
import os
import sys
#sys.path.append("../dataset")

import pandas as pd
from transformers import BertTokenizer,AlbertTokenizer,RobertaTokenizer,AutoTokenizer,DebertaTokenizer
from mutils import remove_punctuation,save_data_to_file,load_data_from_file


def encode_dataset( encoder='bert',max_length_input=512,max_length_sentence=128,max_length_explanation=128):
    cwd = os.getcwd()
    train_tokens_path = cwd + "loaded_data\\train_tokens{}.pkl".format(encoder)

    if not os.path.exists(train_tokens_path):

        file_paths = [cwd + "\\esnli\\esnli_train_1.csv", cwd + "\\esnli\\esnli_train_2.csv"]
        dfs = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        dataset = df.dropna(subset=['Sentence1', 'Sentence2', 'gold_label', 'Explanation_1'], inplace=True)

        if encoder == 'bert':
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if encoder == 'albert':
            tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")

        if encoder == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        if encoder == 'deberta':
            tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

        if encoder == 'tf_transformer':
            tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")

        sentences1 = [remove_punctuation(example['Sentence1']) for example in dataset.to_dict('records')]
        sentences2 = [remove_punctuation(example['Sentence2']) for example in dataset.to_dict('records')]
        explanations = [remove_punctuation(example['Explanation_1']) for example in dataset.to_dict('records')]
        labels = np.array(df["gold_label"].map({"entailment": 0, "neutral": 1, "contradiction": 2})).astype(np.int32)


        encoded_inputs = tokenizer.batch_encode_plus(
              list(zip(sentences1, sentences2)),
              add_special_tokens=True,
              truncation=True,
              max_length=max_length_input,
              padding='max_length',
              return_tensors='tf'
          )
        encoded_s1 = tokenizer.batch_encode_plus(sentences1,
          add_special_tokens=True,
          truncation=True,
          max_length=max_length_sentence,
          padding='max_length',
          return_tensors='tf'

          )

        encoded_s2 = tokenizer.batch_encode_plus(sentences2,
          add_special_tokens=True,
          truncation=True,
          max_length=max_length_sentence,
          padding='max_length',
          return_tensors='tf'

          )

        input_ids_batch = encoded_inputs['input_ids']
        attention_mask_batch = encoded_inputs['attention_mask']
        if encoder != 'roberta':
            token_type_ids_batch = encoded_inputs['token_type_ids']
        else:
            token_type_ids_batch = None
        input_s1_batch = encoded_s1['input_ids']
        input_s2_batch = encoded_s2['input_ids']

        encoded_explanations = tokenizer.batch_encode_plus(
              explanations,
              add_special_tokens=True,
              truncation=True,
              max_length=max_length_explanation,
              padding='max_length',
              return_tensors='tf'
          )

        explanation_input_ids_batch = encoded_explanations['input_ids']
        # explanation_attention_mask_batch = encoded_explanations['attention_mask']
        # explanation_token_type_ids_batch = encoded_explanations['token_type_ids']
        save_data_to_file((input_ids_batch, attention_mask_batch, token_type_ids_batch, input_s1_batch, input_s2_batch, explanation_input_ids_batch,labels), train_tokens_path)
    else:
        input_ids_batch, attention_mask_batch, token_type_ids_batch, input_s1_batch, input_s2_batch, explanation_input_ids_batch,labels = load_data_from_file(train_tokens_path)

    return input_ids_batch, attention_mask_batch, token_type_ids_batch, input_s1_batch, input_s2_batch, explanation_input_ids_batch,labels,tokenizer

#input_ids_batch, attention_mask_batch, token_type_ids_batch, input_s1_batch, input_s2_batch, explanation_input_ids_batch = encode_dataset(encoder = 'bert')
