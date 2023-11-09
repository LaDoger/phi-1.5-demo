import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():

    # Comment this out if you don't have a good Nvidia GPU
    # torch.set_default_device("cuda")

    # Can change your model here
    model_name = 'microsoft/phi-1_5'

    print(f'Loading model: "{model_name}" ...\n')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print('\nModel loaded! Please start prompting!')

    while True:
        prompt = input('----------------\nPrompt: ')

        print('\nInferring...')

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False
        )

        # Change max_length to set output token amount
        outputs = model.generate(**inputs, max_length=32)
        text = tokenizer.batch_decode(outputs)[0]

        print(f'\nOutput: {text}')


if __name__ == '__main__':
    main()