import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

print("Is CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Load the mBLOOM tokenizer and model
generate_model_name = "bigscience/bloom-3b"
generate_tokenizer = BloomTokenizerFast.from_pretrained(generate_model_name)
generate_model = BloomForCausalLM.from_pretrained(generate_model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_model.to(device)

# Check if model on CPU or GPU
print("BLOOM model is on:", next(generate_model.parameters()).device)

# Example input text in English for continuation
input_text = "The spy infiltrated the enemy base to look for the secret documents"

# Tokenize input text
inputs = generate_tokenizer(input_text, return_tensors="pt").to(device)

# Generate text
output_tokens = generate_model.generate(
    inputs["input_ids"], 
    max_length = 2048,
    do_sample = True,
    temperature = 1.0, # higher more creative but less words
    #top_k = 90, # top 70 percent relevant words only
    #top_p = 0.8, # higher the value more creative and diverse
    num_beams = 3, # higher the value less diverse but longer as more sequences to explore
    no_repeat_ngram_size = 2, # room to repeat
    early_stopping = False
)

# Decode and print output text
generated_text = generate_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("Generated text:", generated_text)