import json
import os
import boto3
import base64
import urllib.request
from pinecone import Pinecone

bedrock = boto3.client(service_name='bedrock-runtime', region_name=os.environ.get("MY_AWS_REGION", "us-east-1"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("question", "") 
        image_data = body.get("image", None)
        # MIME TYPE FIX: Accept format from frontend
        image_format = body.get("image_format", "png") 
        
        vision_text = "No visual data provided."

        # --- PHASE 1: NOVA DETAILED VISION ---
        if image_data:
            messages = [{"role": "user", "content": [
                {"image": {"format": image_format, "source": {"bytes": base64.b64decode(image_data)}}},
                {"text": "Perform a highly detailed clinical visual analysis. Describe textures, colors, and any abnormalities in the skin, eyes, or anatomy shown. Be objective."}
            ]}]
            nova_res = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_res['output']['message']['content'][0]['text']

        # --- PHASE 2: PINECONE RAG ---
        search_query = f"{user_query} {vision_text}"
        embed_res = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": search_query.strip() if search_query.strip() else "medical"})
        )
        embedding = json.loads(embed_res['body'].read())['embedding']
        
        index = pc.Index("mediquery")
        pc_res = index.query(vector=embedding, top_k=3, include_metadata=True)
        context_text = "\n\n".join([m['metadata']['text'] for m in pc_res['matches']])

        # --- PHASE 3: THE PROMPT ---
        final_prompt = (
            f"CLINICAL CONTEXT:\n{context_text}\n\n"
            f"VISUAL ANALYSIS:\n{vision_text}\n\n"
            f"USER SYMPTOMS: {user_query}"
        )
        
        req = urllib.request.Request(MODAL_API_URL, method="POST", headers={'Content-Type': 'application/json'})
        modal_res = urllib.request.urlopen(req, data=json.dumps({"prompt": final_prompt}).encode('utf-8'), timeout=120) 
        custom_answer = json.loads(modal_res.read()).get("diagnosis", "Error reading brain output.")

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