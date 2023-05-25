from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import numpy as np
import evaluate
import sys
import argparse
# import wandb
import time

RESULTS_FILE_PATH = './res.txt'
PREDICTIONS_FILE_PATH = './predictions.txt'
TOTAL_TRAINING_TIME = 0
MODEL_DIR_SEED1 = './models/seed1/'
MODEL_DIR_SEED2 = './models/seed2/'


########################################################## HELPERS #####################################################
def tokenize_function(example, tokenizer):
    return tokenizer(example['sentence'], truncation=True)  # NOTE: No fixed padding
def compute_metrics(eval_preds: tuple):
    """
    :param eval_preds: logits outputed by the model
    :return: Accuracy -choice metric in SST2
    """
    metric = evaluate.load('glue', 'sst2')
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)
def train_model(seed: int, checkpoint: str, output_dir: str, train_dataset, validation_dataset, data_collator,
                tokenizer):
    # # Weights & Biases Init:
    # run = wandb.init(
    #     project="ANLP_ex1",
    #     name=checkpoint + "-Seed_number" + str(seed),
    #     tags=[checkpoint, str(seed), "fine_tuning"],
    #     notes="this is a train_run with model: " + checkpoint + " and seed number" + str(seed),
    #     job_type='train'
    # )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    training_args = TrainingArguments(
        evaluation_strategy='no',  # no evaluation during training,
        seed=seed,
        output_dir=output_dir,
        save_total_limit=1  # Only keep the most recent checkpoint
    )  # all rest default
    # NOTE: somewhy save_total_limit = 1 doesnt work on GoogleDrive
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, )
    trainer.train()
    # run.finish()
    return trainer  # TODO: check if need to return

def write_results(checkpoints, models_parameters, res_directory):
    assert len(models_parameters) == len(checkpoints)
    with open('/res.txt', 'a') as f:
        for i in range(len(models_parameters)):
            mean_accuracy = models_parameters[i][0]
            accuracy_std = models_parameters[i][1]
            line = checkpoints[i] + ',' + str(mean_accuracy) + ' +- ' + str(accuracy_std) + '\n'
            f.write(line)
        line = "train time," + str(TOTAL_TRAINING_TIME) + '\n'
        f.write(line)
        f.close()

def evaluate_and_save_model(trainer, model_accuracies, best_seed_output_dir):
    metrics = trainer.evaluate()
    if not model_accuracies or model_accuracies[-1] < metrics['eval_accuracy']:
        trainer.save_model(best_seed_output_dir)
    model_accuracies.append(metrics['eval_accuracy'])
    return metrics['eval_accuracy']

def train_and_evaluate_models(training_args, checkpoints, raw_dataset, best_model_dir):
    best_mean_accuracy = 0
    model_best_trainer = None
    best_model_checkpoint = None
    total_training_time = 0
    models_performance_parameters = []


    for curr_model_checkpoint in checkpoints:
        tokenizer, train_dataset, validation_dataset = load_and_prepare_dataset(training_args, curr_model_checkpoint, raw_dataset)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # model output_dirs to keep tack of best model
        curr_model_dir = './models/current_model/'
        best_seed_output_dir = './models/best_seed/'

        model_accuracies = []
        curr_seed_best_accuracy = 0
        curr_best_trainer = None

        # train each model on each seed
        for seed in range(training_args.seed_num):
            start_time = time.time()
            trainer = train_model(seed, curr_model_checkpoint, curr_model_dir, train_dataset, validation_dataset,
                                  data_collator,
                                  tokenizer)
            end_time = time.time()
            total_training_time += end_time - start_time
            # save only the best seed
            model_accuracy = evaluate_and_save_model(trainer, model_accuracies, best_seed_output_dir)

            if model_accuracy > curr_seed_best_accuracy:
                curr_best_trainer = trainer
                curr_seed_best_accuracy = model_accuracy


        mean_accuracy = np.mean(model_accuracies)
        accuracy_std = np.std(model_accuracies)
        models_performance_parameters.append((mean_accuracy, accuracy_std))
        if mean_accuracy > best_mean_accuracy:
            model_best_trainer = curr_best_trainer
            best_mean_accuracy = mean_accuracy
            best_model_checkpoint = curr_model_checkpoint
    if model_best_trainer:
        model_best_trainer.save_model(best_model_dir)

    return total_training_time, models_performance_parameters,best_model_checkpoint

def load_and_prepare_dataset(args, curr_checkpoint, raw_dataset):
    # pre-processing
    tokenizer = AutoTokenizer.from_pretrained(curr_checkpoint)
    tokenized_dataset = raw_dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    train_dataset = tokenized_dataset['train'].select(range(args.training_count)) if args.training_count != -1 else \
        tokenized_dataset['train']
    validation_dataset = tokenized_dataset['validation'].select(
        range(args.validation_count)) if args.validation_count != -1 else tokenized_dataset['validation']
    return tokenizer, train_dataset, validation_dataset

def write_to_file(path,data,mode='a'):
    with open(path,mode) as f:
        f.write(data)
        f.close()
########################################################################################################################

def main(training_args):
    raw_dataset = load_dataset('sst2')
    checkpoints = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
    best_model_dir = './models/best_model/'
    total_training_time, models_parameters,best_model_checkpoint = train_and_evaluate_models(training_args, checkpoints, raw_dataset,
                                                                       best_model_dir)
    write_results(checkpoints, models_parameters, RESULTS_FILE_PATH)


    # Evaluation on test-set using best model
    tokenizer = AutoTokenizer.from_pretrained(best_model_checkpoint)
    classification_pipeline = pipeline("text-classification", model=best_model_dir, tokenizer=tokenizer)

    test_sentences = raw_dataset['test']['sentence'][:training_args.prediction_count]
    test_labels = raw_dataset['test']['label'][:training_args.prediction_count]

    start_time = time.time()
    predictions = classification_pipeline(test_sentences)
    pred_time = time.time() - start_time
    # write predict time in res.txt
    write_to_file(RESULTS_FILE_PATH,f'predict time {pred_time}\n')
    predicted_labels = [prediction['label'] for prediction in predictions]

    # write preds in prediction.txt
    prediction_text = "\n".join([f"{sentence}###{label[-1]}" for sentence, label in zip(test_sentences,predicted_labels)])
    write_to_file(PREDICTIONS_FILE_PATH,prediction_text)

    if __name__ == "__main__":
        # Command Line Parsing: # assume args are legal values
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed_num', type=int, required=True)
        parser.add_argument('--training_count', type=int, required=True)
        parser.add_argument('--validation_count', type=int, required=True)
        parser.add_argument('--prediction_count', type=int, required=True)
        args = parser.parse_args()
        main(args)
