from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

datasets = load_dataset("oshizo/japanese-wikipedia-paragraphs")

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = datasets.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

model = AutoModelForCausalLM.from_pretrained("rinna/youri-7b-chat")

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset
)

trainer.train()

eval_results = trainer.evaluate()

print(f"Eval results: {eval_results}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_text = "What is Geneleteing AI"
input_data = tokenizer(input_text, return_tensors='pt').to(device)
outputs = model(**input_data)
predicted_class_idx = outputs.logits.argmax(-1).item()
class_dict = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
predicted_class_name = class_dict[predicted_class_idx]
print(f"The input text is classified as: {predicted_class_name}")