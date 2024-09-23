from pip._internal import main as _main
import importlib


name: str = "transformers"
version: str = "4.44.2"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )


name: str = "datasets"
version: str = "3.0.0"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )

    
name: str = "torch"
version: str = "2.4.1"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )



from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

dataset = load_dataset("izumi-lab/llm-japanese-dataset", revision="main")
dataset = load_dataset("izumi-lab/llm-japanese-dataset", revision="a.b.c") # for specific version

tokenizer = AutoTokenizer.from_pretrained("stockmark/stockmark-100b")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

model = AutoModelForCausalLM.from_pretrained("stockmark/stockmark-100b", device_map="auto", torch_dtype=torch.bfloat16)

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset
)

trainer.train()

eval_results = trainer.evaluate()

print(f"Eval results: {eval_results}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_text = "sex"
input_data = tokenizer(input_text, return_tensors='pt').to(device)
outputs = model(**input_data)
predicted_class_idx = outputs.logits.argmax(-1).item()
class_dict = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
predicted_class_name = class_dict[predicted_class_idx]
print(f"The input text is classified as: {predicted_class_name}")