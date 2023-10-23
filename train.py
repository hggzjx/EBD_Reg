import argparse
import os
import tensorflow as tf
from transformers import TFBertModel,TFAlbertModel,TFRobertaModel,TFDebertaModel,TFFunnelModel
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from utils.mutils import save_data_to_file, load_data_from_file
from utils.load_data import load_data
from tensorflow.keras.callbacks import ModelCheckpoint

class train:
    def __init__(self, args, train_on_init=True):
        self.encoder = args.encoder
        self.module = args.module
        self.max_length_input = args.max_length_input
        self.max_length_sentence = args.max_length_sentence
        self.max_length_explanation = args.max_length_explanation
        self.alpha = args.alpha
        self.beta = args.beta
        self.H = args.H
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.optimizer = Adam(2e-5)
        self.additional_losses = args.additional_losses
        self.train_data_path = os.path.join(os.getcwd(), "loaded_data\\train_data_{}.pkl".format(self.encoder))
        self.save_model_path = os.path.join(os.getcwd(), "saved_model\\{}\\{}_weights.h5".format(self.encoder,self.module))

        self.load_data()
        self.build_model()

        if train_on_init:
            self.compile_and_train()

        if self.encoder=='bert':
            self.bert_model=TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        if self.encoder=='albert':
            self.bert_model=TFAlbertModel.from_pretrained("albert-large-v2", output_hidden_states=True, output_attentions=True)
        if self.encoder=='roberta':
            self.bert_model=TFRobertaModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True)
        if self.encoder=='deberta':
            self.bert_model=TFDebertaModel.from_pretrained("microsoft/deberta-base", output_hidden_states=True, output_attentions=True)
        if self.encoder=='funnel_tf':
            self.bert_model=TFFunnelModel.from_pretrained("funnel-transformer/small", output_hidden_states=True, output_attentions=True)

    def load_data(self):
        if not os.path.exists(self.train_data_path):
            self.input_ids_data, self.input_ids_i_data, self.input_ids_j_data, self.weak_labels_data, self.gold_labels_data = load_data(self.encoder, self.max_length_input, self.max_length_sentence, self.max_length_explanation)
        else:
            self.input_ids_data, self.input_ids_i_data, self.input_ids_j_data, self.weak_labels_data, self.gold_labels_data = load_data_from_file(self.train_data_path)

        # self.weak_labels_data = self.weak_labels_data.astype(np.float32)
        # self.gold_labels_data = self.gold_labels_data.astype(np.int32)

    def build_model(self):

        input_ids = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        token_type_ids = layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
        attention_mask_i = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        attention_mask_j = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        input_ids_i = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids_i")
        input_ids_j = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids_j")

        bert_model = self.bert_model
        bert_output = bert_model(input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).last_hidden_state
        bert_output_all_blocks = bert_model(input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).hidden_states
        bert_output_i_all_blocks = bert_model(input_ids_i, attention_mask=attention_mask_i,
                                   token_type_ids=token_type_ids).hidden_states
        bert_output_j_all_blocks = bert_model(input_ids_j, attention_mask=attention_mask_j,
                                   token_type_ids=token_type_ids).hidden_states
        num_classes = 3
        units = bert_output.shape[-1]
        shared_fc1 = layers.Dense(units, activation='tanh')
        shared_fc2 = layers.Dense(1, activation='sigmoid')
        shared_pooling = layers.GlobalMaxPool1D()
        shared_output_layer = layers.Dense(num_classes, activation="softmax")


        bert_output_transformed = shared_fc2(shared_fc1(bert_output))
        bert_output_multiplied = bert_output_transformed * bert_output
        norm = tf.norm(bert_output_multiplied, axis=1, keepdims=True)
        bert_output_multiplied_normalized = bert_output_multiplied / norm
        bert_output_pooled = shared_pooling(bert_output_multiplied_normalized)
        output = shared_output_layer(bert_output_pooled)
        output_baseline = shared_output_layer(bert_output[:, 0, :])

        output_1_all_blocks=[]
        output_i_all_blocks=[]
        output_j_all_blocks=[]

        for bert_output_1, bert_output_i,bert_output_j in zip(bert_output_all_blocks.tolist()[-self.H:],bert_output_i_all_blocks.tolist()[-self.H:],bert_output_j_all_blocks.tolist()[-self.H:]):

            bert_output_1_transformed = shared_fc2(shared_fc1(bert_output_1))
            bert_output_i_transformed = shared_fc2(shared_fc1(bert_output_i))
            bert_output_j_transformed = shared_fc2(shared_fc1(bert_output_j))

            bert_output_1_multiplied = bert_output_1_transformed * bert_output_1
            bert_output_i_multiplied = bert_output_i_transformed * bert_output_i
            bert_output_j_multiplied = bert_output_j_transformed * bert_output_j

            norm_1 = tf.norm(bert_output_1_multiplied, axis=1, keepdims=True)
            bert_output_1_multiplied_normalized = bert_output_1_multiplied / norm_1
            norm_i = tf.norm(bert_output_i_multiplied, axis=1, keepdims=True)
            bert_output_i_multiplied_normalized = bert_output_i_multiplied / norm_i
            norm_j = tf.norm(bert_output_j_multiplied, axis=1, keepdims=True)
            bert_output_j_multiplied_normalized = bert_output_j_multiplied / norm_j

            bert_output_1_pooled = shared_pooling(bert_output_1_multiplied_normalized)
            bert_output_i_pooled = shared_pooling(bert_output_i_multiplied_normalized)
            bert_output_j_pooled = shared_pooling(bert_output_j_multiplied_normalized)

            output_1 = shared_output_layer(bert_output_1_pooled)
            output_i = shared_output_layer(bert_output_i_pooled)
            output_j = shared_output_layer(bert_output_j_pooled)

            output_1_all_blocks.append(output_1)
            output_i_all_blocks.append(output_i)
            output_j_all_blocks.append(output_j)


        bert_output1 = list(bert_model(input_ids).hidden_states)
        attention_scores_all_blocks = []
        for layer_output in bert_output1[-self.H:]:
            attention_scores = tf.matmul(layer_output[:, 0:, :], tf.transpose(layer_output[:, 0:1, :], perm=[0, 2, 1]))
            attention_scores_all_blocks.append(tf.nn.softmax(tf.squeeze(attention_scores, -1), axis=-1))
        attention_scores_all_blocks = tf.stack(attention_scores_all_blocks, axis=-1)

        # build the baseline
        baseline = Model(inputs=input_ids, outputs=output_baseline)

        # build the model
        model = Model(inputs=[input_ids, input_ids_i, input_ids_j],
                      outputs=[output, attention_scores_all_blocks, output_i, output_j])

        if self.module == "baseline":
            self.model = Model(inputs=[input_ids,attention_mask,token_type_ids], outputs=output_baseline)
        if self.module == "ATA":
            self.model = Model(inputs=[input_ids,attention_mask,token_type_ids],outputs=output)
        if self.module == "ATA,EBD_Reg"
            self.model = Model(inputs=[input_ids, input_ids_i, input_ids_j,attention_mask,token_type_ids,attention_mask_i,attention_mask_j], outputs=[output, attention_scores_all_blocks, output_1_all_blocks, output_i_all_blocks, output_j_all_blocks])

    def compile_and_train(self):
        if self.module == "baseline":
            self.model.compile(optimizer=self.optimizer, loss=SparseCategoricalCrossentropy(from_logits=False), metrics='accuracy')
        if self.module == "ATA":
            self.model.compile(optimizer=self.optimizer, loss=SparseCategoricalCrossentropy(from_logits=False), metrics='accuracy')
        if self.module == "ATA,EBD_Reg"
            self.model.compile(optimizer=self.optimizer,
                          loss=lambda y_true, y_pred: self.custom_loss(y_true, y_pred, alpha=self.alpha, beta=self.beta),
                          metrics='accuracy')

        # checkpoint_callback = ModelCheckpoint(
        #     filepath=self.save_model_path+"{}_weights_{epoch:02d}.h5".format(self.module),
        #     save_weights_only=True,
        #     save_best_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     verbose=1
        # )

        history = self.model.fit(
            x=[self.input_ids_data, self.input_ids_i_data, self.input_ids_j_data],
            y=[self.gold_labels_data, self.weak_labels_data, self.weak_labels_data, self.gold_labels_data, self.gold_labels_data],
            epochs=self.epochs,
            batch_size=self.batch_size)

        self.model.save_weights(self.save_model_path)


    def custom_loss(self,y_true, y_pred):

        def js_divergence(y_true, y_pred):
            m = 0.5 * (y_true + y_pred)
            return 0.5 * tf.keras.losses.KLDivergence()(y_true, m) + 0.5 * tf.keras.losses.KLDivergence()(y_pred, m)

        alpha,beta = zip(self.alpha, self.beta)
        y_true_gold = y_true[0]
        y_true_weak = y_true[1]
        y_pred_output = y_pred[0]
        attention_scores_all_blocks = y_pred[1]
        y_pred_output_for_weak_labels = y_pred[2]
        y_pred_output_1_all_blocks = y_pred[3]
        y_pred_output_i_all_blocks = y_pred[4]
        y_pred_output_j_all_blocks = y_pred[5]

        y_true_weak = tf.cast(y_true_weak, tf.float32)

        loss1 = SparseCategoricalCrossentropy()(y_true_gold, y_pred_output)
        y_true_weak = tf.expand_dims(y_true_weak, axis=-1)
        loss2 = tf.reduce_sum(tf.square(attention_scores_all_blocks - y_true_weak), axis=[-1, -2])
        loss3 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true_weak,
                                                                                y_pred_output_for_weak_labels)

        loss4_list = []
        for y_pred_output_1_all_blocks, y_pred_output_i,y_pred_output_j in zip(y_pred_output_1_all_blocks,y_pred_output_i_all_blocks,y_pred_output_j_all_blocks):

            y_pred_output_i_expanded = tf.expand_dims(y_pred_output_i, -1)
            y_pred_output_j_expanded = tf.expand_dims(y_pred_output_j, -1)
            condition_matrix = tf.cast(y_pred_output_i_expanded > y_pred_output_j_expanded, tf.float32)
            product_matrix = y_pred_output_i_expanded * y_pred_output_j_expanded
            q_y = tf.reduce_sum(product_matrix * condition_matrix, axis=1) + tf.reduce_sum(
                product_matrix * (1 - condition_matrix), axis=1)

            loss4_list.append(js_divergence(y_pred_output, q_y))

        loss4 = sum(loss4_list)

        total_loss = loss1
        if 'sa_loss' in self.additional_losses:
            total_loss += beta * loss2
        if 'er_loss' in self.additional_losses:
            total_loss += alpha * loss3
        if 'si_loss' in self.additional_losses:
            total_loss += beta * loss4

        return total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train!")
    parser.add_argument("--encoder", type='str',default="bert")
    parser.add_argument("--module",type=str,default="ATA,EBD_Reg")
    parser.add_argument("--max_length_input", type=int, default=512)
    parser.add_argument("--max_length_sentence", type=int, default=128)
    parser.add_argument("--max_length_explanation", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--H", type=float, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--additional_losses", type=list, default=['sa_loss', 'er_loss', 'si_loss'])

    args = parser.parse_args()
    my_model = train(args)
