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



########################################################################################################################


# Pre-processing
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
raw_dataset = load_dataset('sst2')  # Standford Sentiment Treebank
tokenized_dataset = raw_dataset.map(tokenize_function, batch=True)
model_accuracies = []
# Fine-tuning  model with multiple seeds
for i, seed in enumerate(SEEDS):
    # Weights & Biases Init:
    run = wandb.init(
        project="ANLP_ex1",
        name=checkpoint + "-Seed_number" + str(i),
        tags=[checkpoint,str(i),"fine_tuning"],
        notes = "this is a train_run with model: " + checkpoint + " and seed number" + str(i),
        job_type='train'
    )
    output_dir: str = ""+str(seed)#TODO: set output_dir in collab to drive

    model = AutoModelForSequenceClassification(checkpoint, num_labels=2)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        seed=seed,
    )  # all rest default

    trainer = Trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset['train'][:args.training_count],
        eval_dataset=tokenized_dataset['validation'][:args.validation_count],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    wandb.save(output_dir)
    last_accuracy = trainer.state.log_history[-1]['eval_accuracy']
    model_accuracies.append(last_accuracy)
    run.finish()


#mean and std for the Accuracy of model
mean = np.mean(model_accuracies)
std = np.std(model_accuracies)



# Prediction of best model
model.eval()
# chose best model with best seed
# run prediction with it
predictions = trainer.predict(tokenized_dataset['test'][:args.prediction_count])
