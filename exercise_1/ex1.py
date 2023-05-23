from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
import sys
import argparse
import wandb
TODO:
    -upload train_loss.png 
# Command Line Parsing:
# assume args are legal values
parser = argparse.ArgumentParser()
parser.add_argument('--seed_num', type=int, required=True)
parser.add_argument('--training_count', type=int, required=True)
parser.add_argument('--validation_count', type=int, required=True)
parser.add_argument('--prediction_count', type=int, required=True)
args = parser.parse_args()

SEEDS: list = list(range(args.seed_num))
########################################################################################################################
#model1
#mdoel2
#model3 checkoint/accuracy (inex is seed chosen)
# run following code for each model

########################################################## HELPERS #####################################################
def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True)  # NOTE: No fixed padding


def compute_metrics(eval_preds:tuple):
    """
    :param eval_preds: logits outputed by the model
    :return: Accuracy -choice metric in SST2
    """
    metric = evaluate.load('glue', 'sst2')
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def train_model(seed:int,checkpoint:str,output_dir:str,train_dataset,validation_dataset,data_collator,tokenizer):

    # Weights & Biases Init:
    run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="ANLP_ex1",
        name=checkpoint + "-Seed_number" + str(seed),
        tags=[checkpoint,str(seed),"fine_tuning"],
        notes = "this is a train_run with model: " + checkpoint + " and seed number" + str(seed),
        job_type='train'
    )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    training_args = TrainingArguments(
        evaluation_strategy='no', # no evaluation during training,
        seed=seed,
        report_to = 'wandb',
        output_dir = output_dir,
        save_total_limit = 1 # Only keep the most recent checkpoint
    )  # all rest default
# NOTE: somewhy save_total_limit = 1 doesnt work on GoogleDrive
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,)
    trainer.train()
    run.finish()
    return trainer #TODO: check if need to return


########################################################################################################################

# Checkpoints for each model in Huggingface: bert/roberta/electra
model1_checkpoint = 'bert-base-uncased'
model2_checkpoint = 'roberta-base'
model3_checkpoint = 'google/electra-base-generator'
checkpoint = model1_checkpoint
output_dir: str = '/content/drive/MyDrive/Github/Advance-NLP/exercise_1/models/'+ checkpoint + str(seed)#TODO: set output_dir in collab to drive

# Pre-processing
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
raw_dataset = load_dataset('sst2')  # Standford Sentiment Treebank
tokenized_dataset = raw_dataset.map(tokenize_function,fn_kwargs={'tokenizer':tokenizer}, batched=True)
train_dataset = tokenized_dataset['train'] if training_count == -1 else dataset['train'].select(range(training_count))
validation_dataset = tokenized_dataset['validation'] if validation_count == -1 else dataset['validation'].select(range(validation_count))

# we will run the following code for each model
for seed in SEEDS:
    trainer = train_model(seed,checkpoint,output_dir,train_dataset,validation_dataset,data_collator,tokenizer)
#     train_duration = trainer.state.train_duration

#     # To get the validation inference time, we first need to do an evaluation
#     metrics = trainer.evaluate()
#     eval_duration = trainer.state.eval_duration




#mean and std for the Accuracy of model
mean = np.mean(model_accuracies)
std = np.std(model_accuracies)



# Prediction of best model
model.eval()
# chose best model with best seed
# run prediction with it
predictions = trainer.predict(tokenized_dataset['test'][:args.prediction_count])
