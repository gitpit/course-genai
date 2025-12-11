'''
Docstring for course_work.week5.finetuning03
    This module defines a class for fine-tuning a sequence-to-sequence language model (T5) on a custom instruction-following dataset.
**Important Notes:
 - It works with venv3.11 (python 3.11.9)

'''

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class FineTuneInstructionModel:
    def __init__(self):
        self.model_name = "t5-small"
        self.tokenizer = self.setup_tokenizer()
        self.model = self.setup_model()
        self.dataset = self.load_dataset_dict()
        self.train_dataset = self.create_train_subset()
        self.eval_dataset = self.create_eval_subset()
        self.tokenized_train = self.tokenize_dataset(self.train_dataset)
        self.tokenized_eval = self.tokenize_dataset(self.eval_dataset)
        self.training_args = self.setup_training_args()
        self.trainer = self.setup_trainer()

    def setup_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def setup_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def load_dataset_dict(self):
        data = {
            "instruction": [
                "Summarize the following text in one sentence.",
                "Answer the question based on the text.",
                "Translate the following English text to French.",
                "Extract the main topic from the given paragraph.",
                "Classify the sentiment of this review as positive, negative, or neutral.",
                "Rewrite the sentence to make it more formal.",
                "List three key facts from the provided information.",
                "Convert the following recipe instruction into a question.",
                "Identify the grammatical error in this sentence and correct it.",
                "Paraphrase the following statement using simpler language.",
                "Generate a creative title for this story summary.",
                "Determine whether this claim is true or false based on the context.",
            ],
            "input": [
                "The rain in Spain stays mainly in the plain.",
                "Who is the president of the United States who won the 2024 election?",
                "The cat is sleeping on the warm windowsill.",
                "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations occur naturally, scientific evidence shows that human activities, particularly burning fossil fuels, have been the primary driver of climate change since the 1800s.",
                "This restaurant was absolutely terrible! The food was cold, the service was rude, and we waited over an hour for our meal. I would never recommend this place to anyone.",
                "Hey, can you help me out with this thing?",
                "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "Preheat the oven to 350 degrees Fahrenheit.",
                "The students was excited about there upcoming field trip to the museum.",
                "The individual demonstrated exceptional proficiency in the utilization of advanced computational methodologies.",
                "A young wizard discovers a hidden magical academy where students learn to control the elements, but dark forces threaten to destroy the school and everyone in it.",
                "All birds can fly. Context: Penguins are flightless birds that live in Antarctica.",
            ],
            "output": [
                "Rain in Spain falls in the plain.",
                "Donald Trump.",
                "Le chat dort sur le rebord de fenêtre chaud.",
                "Climate change",
                "Negative",
                "Could you please assist me with this matter?",
                "Python was created by Guido van Rossum in 1991, it emphasizes code readability, and it supports multiple programming paradigms.",
                "What temperature should you preheat the oven to?",
                "The students were excited about their upcoming field trip to the museum.",
                "The person was very good at using advanced computer methods.",
                "Elemental Academy: Rise of the Dark Forces",
                "False",
            ],
        }
        return Dataset.from_dict(data)

    def create_train_subset(self):
        return self.dataset.select(range(1))

    def create_eval_subset(self):
        return self.dataset.select(range(1, 2))

    def preprocess_function(self, examples):
        inputs = [
            f"Instruction: {inst}\nInput: {inp}"
            for inst, inp in zip(examples["instruction"], examples["input"])
        ]
        labels = examples["output"]
        model_inputs = self.tokenizer(inputs, padding="max_length", truncation=True)
        labels = self.tokenizer(labels, padding="max_length", truncation=True)["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    def tokenize_dataset(self, dataset):
        return dataset.map(self.preprocess_function, batched=True)

    def setup_training_args(self):
        return TrainingArguments(
            output_dir="./instruction_result",
            eval_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            num_train_epochs=1,
        )

    def setup_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
        )

    def compute_loss(self, logits, labels):
        shift_logits = logits.view(-1, logits.size(-1)).cpu().numpy()
        shift_labels = labels.view(-1).cpu().numpy()
        num_classes = shift_logits.shape[1]
        batch_size = shift_logits.shape[0]

        labels_one_hot = np.zeros((batch_size, num_classes))
        valid_indices = shift_labels != -100
        labels_one_hot[np.arange(batch_size)[valid_indices], shift_labels[valid_indices]] = 1

        shifted_logits = shift_logits - np.max(shift_logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)

        loss = -np.sum(labels_one_hot * log_probs) / np.sum(valid_indices)
        return loss

    def collate_fn(self, batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch]),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch]),
        }
    
    def evaluate_loss(self):
        model = self.trainer.model
        model.eval()


        eval_dataloader = DataLoader(self.tokenized_eval, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)

                decoder_input_ids = model._shift_right(labels)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                logits = outputs.logits
                loss = self.compute_loss(logits, labels)
                total_loss += loss

        return total_loss / len(eval_dataloader)

    def train_model(self):
        self.trainer.train()


if __name__ == "__main__":
    model_trainer = FineTuneInstructionModel()
    eval_loss_before = model_trainer.evaluate_loss()
    model_trainer.train_model()
    eval_loss_after = model_trainer.evaluate_loss()

    print("It works!!")