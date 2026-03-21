import json
import os
import boto3
import base64
import urllib.request
from pinecone import Pinecone

# Initialize clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Replace with your actual Modal endpoint URL
MODAL_API_URL = "https://skund-kr--mediquery-custom-brain-mediquerybrain-generate-96c12c.modal.run"

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_query = body.get("question", "").strip()
        image_data = body.get("image", None)
        image_format = body.get("image_format", "png")
        
        vision_text = "No visual data provided."
        is_clinical_scan = False

        # --- PHASE 1: THE VISION GATE (Nova Classification) ---
        if image_data:
            messages = [{
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": base64.b64decode(image_data)}}},
                    {"text": "Is this a clinical medical scan (MRI, X-ray, Pathology slide, or professional clinical close-up)? Answer 'YES' or 'NO' and provide a 1-sentence description."}
                ]
            }]
            nova_res = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_res['output']['message']['content'][0]['text']
            is_clinical_scan = "YES" in vision_text.upper()

        # --- PHASE 2: INTENT ROUTING (The 'Anti-Cancer' Filter) ---
        # Detect if the query is "Heavy" (Clinical) or "Light" (Wellness)
        heavy_keywords = ["glaucoma", "syndrome", "tumor", "cancer", "pathology", "diagnosis", "med-"]
        is_heavy_query = any(word in user_query.lower() for word in heavy_keywords) or is_clinical_scan

        # Define source tiers from MedQuad
        # Wellness Mode only looks at general sources like MedlinePlus
        # Clinical Mode opens up specialized sources like NCI (National Cancer Institute)
        filter_logic = {}
        if not is_heavy_query:
            # Metadata filter for Pinecone to only use general sources
            filter_logic = {"source": {"$in": ["MedlinePlus", "NIH", "NIDDK"]}}

        # --- PHASE 3: METADATA-ANCHORED RAG ---
        context_text = "No specific clinical guidelines required for this wellness inquiry."
        
        # Only perform search if there is a clear intent to avoid 'Tired-to-Cancer' drift
        embed_res = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": user_query if user_query else "medical inquiry"})
        )
        embedding = json.loads(embed_res['body'].read())['embedding']
        
        index = pc.Index("mediquery")
        pc_res = index.query(
            vector=embedding, 
            top_k=2, 
            filter=filter_logic, 
            include_metadata=True
        )
        
        # Only use context if it meets a strict similarity threshold
        matches = [m['metadata']['text'] for m in pc_res['matches'] if m['score'] > 0.78]
        if matches:
            context_text = "\n\n".join(matches)

        # --- PHASE 4: DYNAMIC PROMPTING ---
        # We explicitly tell the brain what 'Mode' it is in
        mode = "CLINICAL_DIAGNOSTIC" if is_heavy_query else "GENERAL_WELLNESS"
        
        final_prompt = (
            f"MODE: {mode}\n"
            f"CLINICAL CONTEXT:\n{context_text}\n\n"
            f"VISUAL ANALYSIS:\n{vision_text}\n\n"
            f"USER QUERY: {user_query}\n\n"
            "INSTRUCTION: If MODE is GENERAL_WELLNESS, suggest common causes (rest, flu, stress). "
            "DO NOT suggest terminal illnesses unless MODE is CLINICAL_DIAGNOSTIC."
        )
        
        # --- PHASE 5: BRAIN INFERENCE (Modal) ---
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