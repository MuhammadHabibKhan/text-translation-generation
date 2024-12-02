import gradio as gr
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, BloomTokenizerFast, BloomForCausalLM

print("Is CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Load the mBART model and tokenizer
mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
mbart_tokenizer = MBart50Tokenizer.from_pretrained(mbart_model_name)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

# Load the BLOOM model and tokenizer
bloom_model_name = "bigscience/bloom-3b"
bloom_tokenizer = BloomTokenizerFast.from_pretrained(bloom_model_name)
bloom_model = BloomForCausalLM.from_pretrained(bloom_model_name)

# Move BLOOM model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bloom_model.to(device)

# Move mBART model to GPU if available
mbart_model.to(device)

# Check if model on CPU or GPU
print("mBART model is on:", next(mbart_model.parameters()).device)
print("BLOOM model is on:", next(bloom_model.parameters()).device)

# Supported languages for mBART
languages = {
    "English": "en_XX",
    "French": "fr_XX",
    "Spanish": "es_XX",
    "German": "de_DE",
    "Italian": "it_IT",
    "Portuguese": "pt_XX",
    "Russian": "ru_RU",
    "Chinese": "zh_CN",
    "Japanese": "ja_XX",
    "Arabic": "ar_AR",
}

def translate_and_generate(
    source_text, 
    source_language, 
    target_language, 
    num_beams,
    no_repeat_ngram_size,
    top_k,
    top_p,
    temp
):
    # Set source and target languages for mBART
    mbart_tokenizer.src_lang = languages[source_language]
    input_ids = mbart_tokenizer(source_text, return_tensors="pt").input_ids.to(device)
    
    # Translate text using mBART
    generated_tokens = mbart_model.generate(
        input_ids, 
        forced_bos_token_id=mbart_tokenizer.lang_code_to_id[languages[target_language]]
    )
    translated_text = mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Generate continuation using BLOOM
    inputs = bloom_tokenizer(translated_text, return_tensors="pt").to(device)

    output_tokens = bloom_model.generate(
        inputs["input_ids"], 
        max_length = 2048,
        num_beams = num_beams,
        no_repeat_ngram_size = no_repeat_ngram_size,
        do_sample = True,
        early_stopping = False,
        temperature = temp,
        top_k = top_k,
        top_p = top_p
    )
    generated_text = bloom_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return translated_text, generated_text

# Create the Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("## Text Translation with mBART & Generation with BLOOM 7b")

    with gr.Row():
        gr.Image(value="./Banner.png")
    
    with gr.Row():
        source_text_input = gr.Textbox(label="Input Text", placeholder="Enter the text to translate...")
        source_language_input = gr.Dropdown(choices=list(languages.keys()), label="Source Language", value="French")
        target_language_input = gr.Dropdown(choices=list(languages.keys()), label="Target Language", value="English")

    gr.Markdown("### Hyperparameter Tuning for BLOOM")

    with gr.Row():
        temp_input = gr.Slider(0.0, 1.0, value=1.0, step=1, label="Temperature")
        top_k_input = gr.Slider(0, 100, value=90, step=1, label="Top_K")
        top_p_input = gr.Slider(0.0, 1.0, value=0.8, step=1, label="Top_P")
    
    with gr.Row():
        num_beams_input = gr.Slider(1, 10, value=5, step=1, label="Number of Beams")
        no_repeat_ngram_size_input = gr.Slider(1, 5, value=2, step=1, label="No Repeat N-gram Size")
    
    with gr.Row():
        translate_button = gr.Button("Translate and Generate")
    
    translated_text_output = gr.Textbox(label="Translated Text")
    generated_text_output = gr.Textbox(label="Generated Text")
    
    translate_button.click(
        translate_and_generate, 
        inputs=[
            source_text_input, 
            source_language_input, 
            target_language_input,
            num_beams_input,
            no_repeat_ngram_size_input,
            temp_input,
            top_k_input,
            top_p_input,
        ],
        outputs=[translated_text_output, generated_text_output],
    )

# Launch the demo
demo.launch()
