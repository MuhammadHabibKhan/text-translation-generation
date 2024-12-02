## Overview

Project Name: Text Translation with mBART and Generation using BLOOM
Associated with : Course BSCS-604 Natural Language Processing

## Model Details

### mBART 50
- mBART 50 is a multilingual machine translation model that also uses the transformer architecture. 
- It is trained on a dataset of parallel text in many languages, and is able to translate text between different languages.
- For this project I used the mBART-50 many to one multilingual machine translation fine-tuned check point of the mBART Large 50 pre-trained model by Facebook.
- Find more about it [here](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt)

### BLOOM
- BLOOM is a large language model that uses the transformer architecture to process and generate text.
- It is trained on a massive dataset of text and code, and is capable of generating human-quality text in many languages.
- I have opted for the 7B parameter version for this project
- Find more about BLOOM versions [here](https://huggingface.co/docs/transformers/en/model_doc/bloom).

### Architecture
- mBART first translates the text based on the language selected.
- The translated text is passed onto BLOOM which continues and generates additional text.

![image](https://github.com/user-attachments/assets/142d7f26-8fab-4c9e-a209-b6be8ea13a99)


## Hardware & Performance

Intel Xeon E5-1680v4 | 32GB ECC DDR4 RAM | RTX 2060 6GB | Total GPU memory : 22GB

- The model takes around 14.8GB to 15.5GB of Total GPU Memory at current parameter settings.
- Increasing the number of beams results in larger RAM requirments (21GB for 10 num_beams)
- As my GPU has only 6GB of VRAM, CUDA uses the unified memory from the system.
- This results in decreased performance and much slower computations.
- On average, it takes around 245 seconds (4+ minutes) to get an output on my machine.
- On the contrast, the vanilla BLOOM-560m takes around 2.5GB VRAM and outputs within 20 seconds.
