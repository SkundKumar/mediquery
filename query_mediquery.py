import json
import os
import boto3
import base64
import urllib.request
from pinecone import Pinecone

# --- INITIALIZE CLOUD CLIENTS ---
# Using .get() with defaults prevents the module from crashing on import in CI
bedrock = boto3.client(
    service_name='bedrock-runtime', 
    region_name=os.environ.get("MY_AWS_REGION", "us-east-1")
)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "dummy-key-for-ci"))

# Your Modal GPU endpoint
MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def handler(event, context):
    try:
        # 1. Parse incoming payload
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("question", "") 
        image_data = body.get("image", None)
        image_format = body.get("image_format", "png") # Dynamic MIME detection
        
        vision_text = "No visual data provided."

        # --- PHASE 1: THE EYES (Amazon Nova) ---
        if image_data:
            messages = [{
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": base64.b64decode(image_data)}}},
                    {"text": "Describe the medical findings in this scan clearly. Do not diagnose."}
                ]
            }]
            nova_response = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_response['output']['message']['content'][0]['text']

        # --- PHASE 2: THE MEMORY (Pinecone) ---
        search_text = f"{user_query} {vision_text}" if image_data else user_query
        
        embed_response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": search_text if search_text.strip() else "clinical data"})
        )
        embedding = json.loads(embed_response['body'].read())['embedding']
        
        index = pc.Index("mediquery")
        pc_response = index.query(vector=embedding, top_k=3, include_metadata=True)
        context_text = "\n\n".join([match['metadata']['text'] for match in pc_response['matches']])

        # --- PHASE 3: THE BRAIN (Modal GPU) ---
        # Structured XML prompt to prevent hallucinations and loops
        final_prompt = (
            f"<clinical_guidelines>\n{context_text}\n</clinical_guidelines>\n\n"
            f"<visual_analysis>\n{vision_text}\n</visual_analysis>\n\n"
            f"<patient_question>\n{user_query}\n</patient_question>"
        )
        
        req = urllib.request.Request(MODAL_API_URL, method="POST", headers={'Content-Type': 'application/json'})
        modal_response = urllib.request.urlopen(req, data=json.dumps({"prompt": final_prompt}).encode('utf-8'), timeout=120) 
        custom_answer = json.loads(modal_response.read()).get("diagnosis", "Error reading brain output.")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json", 
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"answer": custom_answer})
        }

    except Exception as e:
        # THE FIX: Added headers back to the error response so pytest passes
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json", 
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }