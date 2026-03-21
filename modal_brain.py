import modal

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
        
        print("Loading Base Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        print("Applying Medical Adapter...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
        print("Brain ready.")

    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        raw_prompt = data.get("prompt", "")
        
        # --- THE HIGH-PRECISION CHAT TEMPLATE ---
        # 1. Clear system role. 2. Proper stop tokens (</s>). 
        # 3. Explicit "I don't know" instruction to prevent hallucinations.
        formatted_prompt = (
            "<|system|>\n"
            "You are a medical assistant. Use the provided context tags to answer the question. "
            "If the information is not in the context, say 'I do not have enough clinical data to answer that.'\n"
            "</s>\n"
            f"<|user|>\n{raw_prompt}</s>\n"
            "<|assistant|>\n"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # FACTUAL SETTINGS: Lower temperature (0.2) + Repetition Penalty
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300,
            do_sample=True,
            temperature=0.2, 
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's answer
        answer = full_text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full_text else full_text
        
        return {"diagnosis": answer}