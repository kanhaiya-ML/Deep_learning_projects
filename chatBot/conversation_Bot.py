# Chat Bot
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

while True:

    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("GoodBye")
        break
    # Enocde input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids,new_input_ids],dim=1)
    else:
        input_ids = new_input_ids

    # generate responses
    response_ids = model.generate(
        input_ids,
        max_length=300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    # Decode Response
    response = tokenizer.decode(
        response_ids[:,input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print(f"Bot: {response}")