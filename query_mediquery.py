import json
import os
import boto3
import base64
import urllib.request
from pinecone import Pinecone

# --- INITIALIZE CLOUD CLIENTS ---
bedrock = boto3.client(service_name='bedrock-runtime', region_name=os.environ.get("MY_AWS_REGION", "us-east-1"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Your Serverless GPU endpoint
MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def handler(event, context):
    try:
        # 1. Parse incoming payload from Streamlit
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("question", "") 
        image_data = body.get("image", None)
        
        vision_text = ""

        # --- PHASE 1: THE EYES (Amazon Nova) ---
        if image_data:
            print("PHASE 1: Extracting visual context...")
            messages = [{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": base64.b64decode(image_data)}}},
                    {"text": "Describe the visual medical anomalies in this image. Do not diagnose."}
                ]
            }]
            nova_response = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_response['output']['message']['content'][0]['text']

        # --- PHASE 2: THE MEMORY (Titan + Pinecone) ---
        print("PHASE 2: Querying Pinecone...")
        search_query = user_query.strip()
        if vision_text:
            search_query += f" [Visual Context: {vision_text}]"
            
        if not search_query.strip():
            search_query = "general clinical medical guidelines"
            
        embed_response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": search_query})
        )
        embedding = json.loads(embed_response['body'].read())['embedding']
        
        # --- PINECONE INDEX NAME (MATCHES YOUR DASHBOARD) ---
        index = pc.Index("mediquery")
        
        pc_response = index.query(vector=embedding, top_k=3, include_metadata=True)
        context_text = "\n".join([match['metadata']['text'] for match in pc_response['matches']])

        # --- PHASE 3: THE BRAIN (Modal GPU) ---
        print("PHASE 3: Routing data to Custom Modal Brain...")
        final_prompt = f"Guidelines:\n{context_text}\n\nVisual Analysis:\n{vision_text}\n\nUser Question:\n{user_query}\n\nDiagnosis/Answer:"
        
        req = urllib.request.Request(MODAL_API_URL, method="POST", headers={'Content-Type': 'application/json'})
        
        # INCREASED TIMEOUT: From 25 to 60 seconds to handle GPU cold starts
        modal_response = urllib.request.urlopen(req, data=json.dumps({"prompt": final_prompt}).encode('utf-8'), timeout=60) 
        
        custom_answer = json.loads(modal_response.read()).get("diagnosis", "Error reading custom brain output.")

        # --- RETURN ---
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"answer": custom_answer})
        }

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": f"Internal Server Error: {str(e)}"})
        }