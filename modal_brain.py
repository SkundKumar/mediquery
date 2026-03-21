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
        print("Brain Ready.")

    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        import torch
        raw_prompt = data.get("prompt", "")
        
        # System instructions to enforce brevity and prevent repetition
        system_instr = "You are a direct medical assistant. Answer immediately. Do not repeat the user's input."
        
        # The 'Force-Start' technique for TinyLlama
        formatted_prompt = (
            f"<|system|>\n{system_instr}</s>\n"
            f"<|user|>\n{raw_prompt}</s>\n"
            f"<|assistant|>\nThe most likely cause is"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.2, 
                repetition_penalty=1.3 
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parsing logic to extract only the assistant's new response
        if "<|assistant|>" in full_text:
            answer = full_text.split("<|assistant|>")[-1].strip()
        else:
            answer = full_text.replace(formatted_prompt, "").strip()

        # Final formatting cleanup
        if not answer.lower().startswith("the most likely cause is"):
            answer = "The most likely cause is " + answer
            
        return {"diagnosis": answer}