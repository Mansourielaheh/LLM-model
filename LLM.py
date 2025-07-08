## health-related questions using a pre-trained LLM.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
# Example:
question = "What are the symptoms of diabetes?"
context = "Common symptoms of diabetes include frequent urination, increased thirst, and unexplained weight loss."
input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_length=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
