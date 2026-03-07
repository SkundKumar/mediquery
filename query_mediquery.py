import json
import os
import boto3
from pinecone import Pinecone
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

def handler(event, context):
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("mediquery")
        
        region = os.getenv("MY_AWS_REGION", "us-east-1")
        bedrock_boto = boto3.client(service_name="bedrock-runtime", region_name=region)

        # Handle both API Gateway/Function URL formats and direct CLI invokes
        body_str = event.get("body", "{}")
        if isinstance(body_str, dict):
            body = body_str
        else:
            body = json.loads(body_str)
            
        question = body.get("question", "What is Glaucoma?")

        # 1. Embed using Titan V2
        emb_res = bedrock_boto.invoke_model(
            modelId="amazon.titan-embed-text-v2:0", 
            body=json.dumps({
                "inputText": question,
                "dimensions": 1024,
                "normalize": True
            })
        )
        query_embedding = json.loads(emb_res['body'].read())['embedding']

        # 2. Query Pinecone
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context_text = "\n".join([res['metadata']['text'] for res in results['matches']])

        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer based only on the context provided:"
        
        # 3. Generate Answer using Native Nova Lite
        gen_res = bedrock_boto.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{
                "role": "user",
                "content": [{"text": prompt}]
            }],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.3
            }
        )
        
        answer = gen_res['output']['message']['content'][0]['text']

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }