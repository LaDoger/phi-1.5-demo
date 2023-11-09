import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():

    # Comment this out if you don't have a good Nvidia GPU
    torch.set_default_device('cuda')

    # Can change your model here
    model_name = 'microsoft/phi-1_5'

    print(f'Loading model: "{model_name}" ...\n')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    print('\nModel loaded! Please start prompting!')

    def infer(prompt, max_length) -> str:
        """Returns predicted words AFTER the prompt text"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False
        )

        outputs = model.generate(**inputs, max_length=max_length)
        text = tokenizer.batch_decode(outputs)[0]

        return text[len(prompt):]
    

    while True:
        prompt = input('----------------\nPrompt: ')
        prompt_token_count = len(tokenizer.tokenize(prompt))
        
        print(f'\nOutput: {prompt}', end='')

        # Change this to change how many tokens you want to generate
        for i in range(100): # Generate 100 tokens

            next_word = infer(prompt, max_length=prompt_token_count + 1)
            print(next_word, end='', flush=True)
            
            prompt += next_word
            prompt_token_count += 1
        
        print('')
            

if __name__ == '__main__':
    main()