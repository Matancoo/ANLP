from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
import sys
import argparse
import wandb
import time
# TODO:
#     -upload train_loss.png 

RESULTS_FILE_PATH = '/res.txt'
PREDICTIONS_FILE_PATH = '/predictions.txt'
TOTAL_TRAINING_TIME = 0


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

def write_results(checkpoints,models_parameters,res_directory):
    assert len(models_parameters) == len(checkpoints)
    with open('/res.txt','a') as f:
        for i in range(len(models_parameters)):
            mean_accuracy = models_parameters[i][0]
            accuracy_std = models_parameters[i][1]
            line = checkpoints[i] + ',' + str(mean_accuracy) + '+-' + str(accuracy_std) + '\n'
            f.write(line)
         line = "train time," + str(TOTAL_TRAINING_TIME) + '\n'
         f.write(line)
         f.close()
########################################################################################################################

def main(args):
    raw_dataset = load_dataset('sst2')  # Standford Sentiment Treebank
    tokenized_dataset = raw_dataset.map(tokenize_function,fn_kwargs={'tokenizer':tokenizer}, batched=True)
    train_dataset = tokenized_dataset['train'] if training_count == -1 else dataset['train'].select(range(training_count))
    validation_dataset = tokenized_dataset['validation'] if validation_count == -1 else dataset['validation'].select(range(validation_count))
    # Checkpoints for models 
    checkpoints = ['bert-base-uncased','roberta-base','google/electra-base-generator']
#     best_model_output_dir = ''
#     best_checkpoint = ''
#     best_seed_model = ''
    models_parameters = [] # tupple of (mean_accuracy,accuracy_std) for each model


    # train each model on each seed and record metrics
    for checkpoint in checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model_accuracies = []
        for seed in range(args.seed_num)):
            start_time = time.time()
            
            if best_seed_model:
                
            trainer = train_model(seed,checkpoint,output_dir,train_dataset,validation_dataset,data_collator,tokenizer)
            end_time = time.time()
            training_time = end_time-start_time
            TOTAL_TRAINING_TIME += training_time 
            metrics = trainer.evaluate()
            model_accuracies.append(metrics['eval_accuracy'])

        # calculating mean & std of accuracies of models trained on different seeds
        mean_accuracy = np.mean(model_accuracies)
        accuracy_std = np.std(model_accuracies)
        models_parameters.append(mean_accuracy,accuracy_std)
        
        # compaire models to save only the best 
#         if best_model_output_dir:
            


    # updating res.txt as specified in Exercise
    write_results(checkpoints,models_parameters'./res.txt')

    # Evaluation on testset using best model
    test_dataset = raw_dataset['test]
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    classification_pipeline = pipeline("text-classification", model=best_model_output_dir, tokenizer=tokenizer)
    start_time = time.time() 
                              
    test_sentences = test_dataset['sentence']
    test_labels = test_dataset['label']
                               
    predictions = classification_pipeline(test_sentences)
    end_time = time.time()
    pred_time = end_time-start_time                           
    with open('path','a') as f:
                               
    predicted_labels = [prediction['label'] for prediction in predictions]
    # TOSO: get accuuracy
    # writing perdiction.txt as specified in exercise
    with open('path','a') as f:
      for tup in list(zip(test_sentences,predicted_labels)):
        line = tup[0] + "###" + tup[1][-1] + "\n"
        f.write(line)
      f.close


    if __name__ == "__main__":
    # Command Line Parsing: # assume args are legal values
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_num', type=int, required=True)
    parser.add_argument('--training_count', type=int, required=True)
    parser.add_argument('--validation_count', type=int, required=True)
    parser.add_argument('--prediction_count', type=int, required=True)
    args = parser.parse_args()
    main(args)

        
    

                            # bert-base-uncased,<mean accuracy> +- <accuracy std>
# roberta-base,<mean accuracy> +- <accuracy std>
# google/electra-base-generator,<mean accuracy> +- <accuracy std>
# ----
# train time,19263
# predict time,<predict time in seconds>
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
