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
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
        self.model = PeftModel.from_pretrained(base, ADAPTER_MODEL)

    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        raw_prompt = data.get("prompt", "")
        
        # --- THE COMMON SENSE SYSTEM PROMPT ---
        system_instr = (
            "You are a grounded medical assistant for general 'feel weird' inquiries. "
            "1. Prioritize common conditions (cold, flu, allergies) for simple symptoms. "
            "2. If the clinical guidelines provided seem too extreme for the symptoms, mention the common possibility first. "
            "3. If an image is provided but isn't a medical scan, describe it simply and do not give a clinical diagnosis."
        )

        formatted_prompt = f"<|system|>\n{system_instr}</s>\n<|user|>\n{raw_prompt}</s>\n<|assistant|>\n"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3, # Low temperature for factual consistency
            repetition_penalty=1.2 # Stops the 'asda' looping
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"diagnosis": full_text.split("<|assistant|>")[-1].strip()}