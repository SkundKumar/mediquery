import modal

# 1. Name your cloud application
app = modal.App("mediquery-custom-brain")

# 2. Define the Docker environment (Now explicitly including fastapi[standard])
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", 
    "transformers", 
    "accelerate",
    "fastapi[standard]"
)

# 3. Define your model location
MODEL_NAME = "skunddd/mediquery-tinyllama-v1"

# 4. Attach a T4 GPU and wrap the deployment in a class (Updated scaledown syntax)
@app.cls(image=image, gpu="T4", scaledown_window=120)
class MediQueryBrain:
    
    # This runs ONCE when the GPU boots up (Cold Start)
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading custom weights into VRAM...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        print("Model loaded successfully.")

    # This creates the live API URL (Updated FastAPI syntax)
    @modal.fastapi_endpoint(method="POST")
    def generate_diagnosis(self, data: dict):
        prompt = data.get("prompt", "No prompt provided.")
        
        # Format the context and question
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate the medical answer
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"diagnosis": answer}