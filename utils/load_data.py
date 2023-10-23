import os
import tensorflow as tf
import numpy as np
import sys
from encode_data import encode_dataset
from mutils import save_data_to_file,load_data_from_file

def load_data(encoder = 'bert',max_length_input=512,max_length_sentence=128,max_length_explanation=128):
    cwd = os.getcwd()
    train_data_path = cwd + "loaded_data\\train_data_{}.pkl".format(encoder)
    if not os.path.exists(train_data_path):


        input_ids_batch, attention_mask_batch, token_type_ids_batch, input_s1_batch, input_s2_batch, explanation_input_ids_batch,labels,tokenizer = encode_dataset(encoder,max_length_input,max_length_sentence,max_length_explanation)

        weak_labels_batch = []
        weak_labels_norm = []
        attention_mask_i_batch = []
        attention_mask_j_batch = []
        index = 0
        for input_ids, explanation_input_ids, s1, s2 in zip(input_ids_batch, explanation_input_ids_batch, input_s1_batch, input_s2_batch):
            attention_mask_i=[]
            attention_mask_j=[]
            weak_label = []

            tokens1 = tokenizer.convert_ids_to_tokens(input_ids)
            tokens2 = tokenizer.convert_ids_to_tokens(explanation_input_ids)
            tokens_s1 = tokenizer.convert_ids_to_tokens(s1)
            tokens_s2 = tokenizer.convert_ids_to_tokens(s2)

            intersection = set(tokens_s1) & set(tokens_s2)

            for token in tokens1:
                if token =='[PAD]':
                  attention_mask_i.append(0)
                  attention_mask_j.append(0)
                  weak_label.append(0)
                elif token =='[CLS]' or token=='[SEP]':
                  attention_mask_i.append(1)
                  attention_mask_j.append(1)
                  weak_label.append(0)
                elif token in tokens2:
                  attention_mask_i.append(0)
                  attention_mask_j.append(1)
                  if token in intersection:
                    weak_label.append(1)
                  else:
                    weak_label.append(2)
                else:
                  attention_mask_i.append(1)
                  attention_mask_j.append(0)
                  weak_label.append(0)

            norm_factor = sum(weak_label)
            if norm_factor == 0:
              norm_factor += 1e-5
            else:
              pass
            weak_label_norm = [w / norm_factor for w in weak_label]
            weak_labels_batch.append(weak_label)
            weak_labels_norm.append(weak_label_norm)

            attention_mask_i_batch.append(attention_mask_i)
            attention_mask_j_batch.append(attention_mask_j)
            index += 1
            print('\r{} sentence pairs have been encoded'.format(index),end='')


        weak_labels_batch = np.array(weak_labels_batch).astype(np.float32)
        weak_labels_norm = np.array(weak_labels_norm).astype(np.float32)
        attention_mask_i_batch = tf.constant(attention_mask_i_batch, dtype=tf.int32)
        attention_mask_j_batch = tf.constant(attention_mask_j_batch, dtype=tf.int32)

        save_data_to_file((input_ids_batch, attention_mask_batch, token_type_ids_batch, attention_mask_i_batch, attention_mask_j_batch, labels,
                           weak_labels_batch, weak_labels_norm), train_data_path)

    else:
        input_ids_batch, attention_mask_batch, token_type_ids_batch, attention_mask_i_batch, attention_mask_j_batch, labels,
        weak_labels_batch, weak_labels_norm = load_data_from_file(
            train_data_path)

    return (
        input_ids_batch,
        attention_mask_batch,
        token_type_ids_batch,
        attention_mask_i_batch,
        attention_mask_j_batch,
        labels,
        weak_labels_batch,
        weak_label_norm
    )