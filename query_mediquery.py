import json
import os
import boto3
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

def handler(event, context):
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("mediquery")
        
        region = os.getenv("MY_AWS_REGION", "us-east-1")
        
        # Boto3 Client for the Titan Embedding Model
        bedrock_boto = boto3.client(service_name="bedrock-runtime", region_name=region)

        # OpenAI-Compatible Client for the Mantle Endpoint
        # Dynamically pulling the exact Bearer token from Lambda environment variables
        bedrock_openai = OpenAI(
            api_key=os.getenv("AWS_BEARER_TOKEN_BEDROCK"), 
            base_url=f"https://bedrock-mantle.{region}.api.aws/v1" 
        )

        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "What is Glaucoma?")

        # --- STEP 1: EMBEDDING (Titan V2 via Boto3 - Proven Working) ---
        emb_res = bedrock_boto.invoke_model(
            modelId="amazon.titan-embed-text-v2:0", 
            body=json.dumps({
                "inputText": question,
                "dimensions": 1024,
                "normalize": True
            })
        )
        query_embedding = json.loads(emb_res['body'].read())['embedding']

        # --- STEP 2: PINECONE ---
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context_text = "\n".join([res['metadata']['text'] for res in results['matches']])

        # --- STEP 3: GENERATION (via OpenAI-compatible API) ---
        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer based only on the context provided:"
        
        # Using the supported model ID for the Mantle endpoint
        response = bedrock_openai.chat.completions.create(
            model="openai.gpt-oss-120b", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )
        
        answer = response.choices[0].message.content

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }