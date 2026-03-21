import modal
import os

app = modal.App("mediquery-custom-brain")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "transformers", "accelerate", "peft", "fastapi[standard]"
)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_MODEL = "skunddd/mediquery-tinyllama-v1"

@app.cls(image=image, gpu="T4", scaledown_window=120)
class MediQueryBrain:
    
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print("LOADING: Tokenizer and Base Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        print(f"LOADING: Adapter {ADAPTER_MODEL}...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
        print("SUCCESS: Brain is online.")

    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        import torch
        raw_prompt = data.get("prompt", "")
        
        # Aggressive direct prompt to stop the 'parrot' behavior
        system_instr = "You are a direct medical assistant. Answer immediately. Do not repeat the user."
        formatted_prompt = f"<|system|>\n{system_instr}</s>\n<|user|>\n{raw_prompt}</s>\n<|assistant|>\nThe most likely cause is"

        # FIXED: Explicitly getting input_ids as a tensor to avoid the 'list' error
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        print("INFERENCE: Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.2, 
                repetition_penalty=1.3 
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's answer and cleanup
        answer = full_text.split("<|assistant|>")[-1].strip()
        if not answer.lower().startswith("the most likely cause is"):
            answer = "The most likely cause is " + answer
            
        return {"diagnosis": answer}