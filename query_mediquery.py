import json
import os
import boto3
import base64
import urllib.request
from pinecone import Pinecone

# --- INITIALIZE CLOUD CLIENTS ---
bedrock = boto3.client(service_name='bedrock-runtime', region_name=os.environ.get("MY_AWS_REGION", "us-east-1"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("question", "") 
        image_data = body.get("image", None)
        
        vision_text = ""

        # PHASE 1: Nova Vision Extraction
        if image_data:
            messages = [{"role": "user", "content": [
                {"image": {"format": "png", "source": {"bytes": base64.b64decode(image_data)}}},
                {"text": "Describe the visual medical anomalies in this image."}
            ]}]
            nova_response = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_response['output']['message']['content'][0]['text']

        # PHASE 2: Pinecone Retrieval (Index: mediquery)
        search_query = user_query.strip()
        if vision_text: search_query += f" [Visual Context: {vision_text}]"
        if not search_query.strip(): search_query = "medical clinical guidelines"
            
        embed_response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": search_query})
        )
        embedding = json.loads(embed_response['body'].read())['embedding']
        
        # MATCHES YOUR PINECONE DASHBOARD
        index = pc.Index("mediquery")
        pc_response = index.query(vector=embedding, top_k=3, include_metadata=True)
        context_text = "\n".join([match['metadata']['text'] for match in pc_response['matches']])

        # PHASE 3: Modal Brain Call
        final_prompt = f"Guidelines:\n{context_text}\n\nVision:\n{vision_text}\n\nQuestion:\n{user_query}\n\nDiagnosis:"
        
        req = urllib.request.Request(MODAL_API_URL, method="POST", headers={'Content-Type': 'application/json'})
        # 120s timeout to allow for the model loading
        modal_response = urllib.request.urlopen(req, data=json.dumps({"prompt": final_prompt}).encode('utf-8'), timeout=120) 
        custom_answer = json.loads(modal_response.read()).get("diagnosis", "Error reading brain output.")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"answer": custom_answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }