import torch

def normalize_text(tokenizer, model, user_input):
    inputs = tokenizer(user_input, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.2,
            max_length=len(user_input),
            num_beams=10,
            early_stopping=True
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    return decoded_output
