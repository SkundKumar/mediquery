import json
import os
import boto3
import time
import logging
import base64
from pinecone import Pinecone
from dotenv import load_dotenv

# 1. Initialize CloudWatch Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.exists(".env"):
    load_dotenv()

def handler(event, context):
    request_start_time = time.time()
    logger.info("INCOMING REQUEST: Initializing Mediquery Inference...")
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("mediquery")
        region = os.getenv("MY_AWS_REGION", "us-east-1")
        bedrock_boto = boto3.client(service_name="bedrock-runtime", region_name=region)

        # Parse Payload
        body_str = event.get("body", "{}")
        body = body_str if isinstance(body_str, dict) else json.loads(body_str)
            
        question = body.get("question", "What is Glaucoma?")
        image_base64 = body.get("image", None) # Optional image payload!

        # --- TELEMETRY: Vector Retrieval ---
        retrieval_start = time.time()
        
        emb_res = bedrock_boto.invoke_model(
            modelId="amazon.titan-embed-text-v2:0", 
            body=json.dumps({"inputText": question, "dimensions": 1024, "normalize": True})
        )
        query_embedding = json.loads(emb_res['body'].read())['embedding']
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
        context_text = "\n".join([res['metadata']['text'] for res in results['matches']])
        
        retrieval_time = round((time.time() - retrieval_start) * 1000, 2)
        logger.info(f"PINECONE RETRIEVAL: {retrieval_time}ms | Context fetched.")

        # --- MULTIMODAL BEDROCK INFERENCE ---
        gen_start = time.time()
        
        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer based on the context and the provided image (if any):"
        
        # Build the message block dynamically
        message_content = [{"text": prompt}]
        
        # If the frontend sent an image, inject it into the AI's vision cortex
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            message_content.append({
                "image": {
                    "format": "jpeg", # Nova supports jpeg, png, webp
                    "source": {"bytes": image_bytes}
                }
            })
            logger.info("MULTIMODAL: Vision payload detected and injected.")

        gen_res = bedrock_boto.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": message_content}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.3}
        )
        
        answer = gen_res['output']['message']['content'][0]['text']
        
        # --- TELEMETRY: Generation & Total Time ---
        gen_time = round((time.time() - gen_start) * 1000, 2)
        total_time = round((time.time() - request_start_time) * 1000, 2)
        logger.info(f"BEDROCK GENERATION: {gen_time}ms | TOTAL LATENCY: {total_time}ms")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        logger.error(f"CRITICAL FAILURE: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }