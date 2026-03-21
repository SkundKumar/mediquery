import modal

app = modal.App("mediquery-custom-brain")

# Added 'peft' to requirements to handle your adapter files
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", 
    "transformers", 
    "accelerate",
    "peft",
    "fastapi[standard]"
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
            BASE_MODEL, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        print("Applying Medical Fine-Tuning Adapter...")
        # This combines the base brain with your specific medical 'patch'
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
        print("Custom Brain loaded successfully.")

    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        prompt = data.get("prompt", "")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the output so it doesn't repeat the prompt
        clean_answer = answer.replace(prompt, "").strip()
        return {"diagnosis": clean_answer}