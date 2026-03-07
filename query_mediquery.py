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
        
        # We now use the native boto3 client for EVERYTHING
        bedrock_boto = boto3.client(service_name="bedrock-runtime", region_name=region)

        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "What is Glaucoma?")

        # 1. Embed the Question (Using Titan V2)
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

        # 3. Generate Answer (Using Native Titan Text Express)
        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer based only on the context provided:"
        
        gen_res = bedrock_boto.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.3
                }
            })
        )
        answer_data = json.loads(gen_res['body'].read())
        answer = answer_data['results'][0]['outputText']

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