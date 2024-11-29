import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

print("Is CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Example with mBART
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Move BLOOM model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check if model on CPU or GPU
print("mBART model is on:", next(model.parameters()).device)

# Set the source and target languages
source_text = "Le flash active sa force de vitesse et se déplace à Mach 2, courant désormais sur l'eau de la rivière. Il le fait pour sauver la ville en arrêtant un métahumain qui allait exploser."

tokenizer.src_lang = "fr_XX"
input_ids = tokenizer(source_text, return_tensors="pt").input_ids.to(device)

# Translate to French
generated_tokens = model.generate(input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print("Translated text:", translated_text)