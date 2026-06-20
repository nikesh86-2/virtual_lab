"""
evaluate_model.py

Simple prompt test for your trained model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "training/output_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


def run_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_prompt = "Evaluate RNA structural stability and suggest design improvements."
    run_prompt(test_prompt)