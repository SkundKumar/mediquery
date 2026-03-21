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
        user_query = body.get("question", "").lower()
        image_data = body.get("image", None)
        image_format = body.get("image_format", "png")
        
        is_medical_scan = False
        vision_text = "No image provided."

        # --- PHASE 1: THE GATEKEEPER (Nova Classification) ---
        if image_data:
            print("PHASE 1: Classifying image...")
            messages = [{
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": base64.b64decode(image_data)}}},
                    {"text": "Classify this image: Is it a (A) Professional Medical Scan (X-ray/MRI), (B) A casual photo of a body part (skin/face), or (C) A non-medical object/person? Describe the visual facts only."}
                ]
            }]
            nova_response = bedrock.converse(modelId="amazon.nova-lite-v1:0", messages=messages)
            vision_text = nova_response['output']['message']['content'][0]['text']
            
            # If Nova sees a selfie or non-medical, we flag it
            if "professional medical scan" in vision_text.lower():
                is_medical_scan = True

        # --- PHASE 2: CONTEXTUAL PRE-PROCESSING ---
        # If the user is just 'tired' and it's a selfie, we don't want cancer data.
        # We only search Pinecone if the query contains 'heavy' medical keywords.
        medical_keywords = ["pain", "glaucoma", "fever", "infection", "growth", "vision", "syndrome"]
        is_heavy_query = any(word in user_query for word in medical_keywords) or is_medical_scan

        context_text = "No relevant clinical guidelines found for this common inquiry."
        
        if is_heavy_query:
            print("PHASE 2: Fetching targeted clinical context...")
            embed_res = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": f"{user_query} {vision_text}"})
            )
            embedding = json.loads(embed_res['body'].read())['embedding']
            
            index = pc.Index("mediquery")
            # We filter for high confidence scores only (0.8+)
            pc_res = index.query(vector=embedding, top_k=2, include_metadata=True)
            
            matches = [m['metadata']['text'] for m in pc_res['matches'] if m['score'] > 0.82]
            if matches:
                context_text = "\n\n".join(matches)

        # --- PHASE 3: DYNAMIC PROMPTING ---
        # We tell the brain exactly how to behave based on our classification
        behavior_mode = "DIAGNOSTIC" if is_medical_scan else "GENERAL_WELLNESS"
        
        final_prompt = (
            f"MODE: {behavior_mode}\n"
            f"CLINICAL GUIDELINES:\n{context_text}\n\n"
            f"VISUAL OBSERVATION:\n{vision_text}\n\n"
            f"USER QUESTION: {user_query}\n\n"
            "INSTRUCTION: If MODE is GENERAL_WELLNESS, avoid terminal diagnoses like cancer or rare syndromes. "
            "Focus on common causes like rest, hydration, or stress unless a scan is provided."
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
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}