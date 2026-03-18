import json
import os
import boto3
import base64
import urllib.request
import urllib.error
from pinecone import Pinecone

# --- INITIALIZE CLOUD CLIENTS ---
bedrock = boto3.client(service_name='bedrock-runtime', region_name=os.environ.get("MY_AWS_REGION", "us-east-1"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Your new Serverless GPU endpoint
MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def lambda_handler(event, context):
    try:
        # 1. Parse incoming payload from Streamlit
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("query", "")
        image_data = body.get("image", None)
        
        vision_text = ""

        # --- PHASE 1: THE EYES (Amazon Nova Vision Extraction) ---
        if image_data:
            print("PHASE 1: Extracting visual context with Nova Lite...")
            messages = [{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": base64.b64decode(image_data)}}},
                    {"text": "You are a clinical data extractor. Describe every visual medical anomaly in this image in plain text. Do not diagnose. Just describe what you see."}
                ]
            }]
            nova_response = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_response['output']['message']['content'][0]['text']
            print(f"Vision Extracted: {vision_text}")

        # --- PHASE 2: THE MEMORY (Titan Embeddings + Pinecone) ---
        print("PHASE 2: Querying Pinecone with enriched context...")
        # Dynamically enrich the vector search if an image was uploaded
        search_query = user_query
        if vision_text:
            search_query += f" [Visual Context: {vision_text}]"
            
        # Get Titan Embedding
        embed_payload = json.dumps({"inputText": search_query})
        embed_response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=embed_payload
        )
        embedding = json.loads(embed_response['body'].read())['embedding']
        
        # Search Pinecone
        index = pc.Index("mediquery-index")
        pc_response = index.query(vector=embedding, top_k=3, include_metadata=True)
        
        context_text = "\n".join([match['metadata']['text'] for match in pc_response['matches']])
        print("Context retrieved from Pinecone.")

        # --- PHASE 3: THE BRAIN (Custom TinyLlama via Modal API) ---
        print("PHASE 3: Routing data to Custom Modal Brain...")
        
        # Package the ultimate prompt for your fine-tuned model
        final_prompt = f"Clinical Guidelines:\n{context_text}\n\nVisual Analysis:\n{vision_text}\n\nUser Question:\n{user_query}\n\nDiagnosis/Answer:"
        
        # Fire the HTTP POST request to your Modal T4 GPU
        req = urllib.request.Request(MODAL_API_URL, method="POST", headers={'Content-Type': 'application/json'})
        data = json.dumps({"prompt": final_prompt}).encode('utf-8')
        
        # 25 second timeout to allow the T4 GPU to boot up from zero
        modal_response = urllib.request.urlopen(req, data=data, timeout=25) 
        modal_result = json.loads(modal_response.read())
        
        final_answer = modal_result.get("diagnosis", "Error reading custom brain output.")
        print("Brain synthesis complete.")

        # --- RETURN TO STREAMLIT ---
        return {
            "statusCode": 200,
            "body": json.dumps({"response": final_answer})
        }

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal Server Error: {str(e)}"})
        }