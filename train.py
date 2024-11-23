from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset


dataset = load_dataset('heegyu/toxic-spans')

models = ['lyeonii/bert-tiny',
          'lyeonii/bert-small',
          'lyeonii/bert-medium',
          'google-bert/bert-base-uncased',
          'google-bert/bert-large-uncased',
          'smallbenchnlp/roberta-small',
          'JackBAI/roberta-medium',
          'FacebookAI/roberta-base',
          'FacebookAI/roberta-large']


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)


for model in models:
    print(f"Fine-tuning model: {model}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForTokenClassification.from_pretrained(model, num_labels=2)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    output_dir = f"./results/{model}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model {model} fine-tuned and saved to {output_dir}")