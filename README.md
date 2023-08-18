# EBD_Reg
paper: EBD-Reg: Explanation based Bias Decoupling Regularization
for Natural Language Inference
# Environment
+ python3.10
+ tensorflow2.10
+ cuda11.6
+ transformers4.17
+ RTX3090

# Train the model:
python train.py


# Train with your data
Process your own dataset into a csv file (in the evaluation folder) with three fields for header Sentence1, Sentence2, gold_label 

python evaluate.py
