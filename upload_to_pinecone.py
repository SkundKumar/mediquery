import os
import json
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm
import time

# 1. Setup - Reading your secret keys
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("mediquery")
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# The path to your medical data
DATA_PATH = "data/med_data_cleaned.jsonl" 

def get_embedding(text):
    """Turns medical text into 1024 numbers using Amazon Bedrock."""
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024, 
        "normalize": True
    })
    try:
        response = bedrock.invoke_model(
            body=body, 
            modelId="amazon.titan-embed-text-v2:0", 
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")
    except Exception as e:
        print(f"\n❌ AWS Error on text: {text[:30]}... | Error: {e}")
        return None

def upload_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: I can't find {DATA_PATH}. Please check your 'data' folder.")
        return

    print("🚀 Starting Medical Data Upload (This will take ~45-60 minutes)...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    batch_size = 50 
    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i:i + batch_size]
        to_upsert = []
        
        for j, line in enumerate(batch_lines):
            try:
                data = json.loads(line)
                vector = get_embedding(data['text'])
                
                if vector:
                    to_upsert.append({
                        "id": f"med-{i+j}",
                        "values": vector,
                        "metadata": {"text": data['text']}
                    })
            except Exception as e:
                continue
            
        if to_upsert:
            index.upsert(vectors=to_upsert)
            time.sleep(0.1) # Prevents AWS from getting overwhelmed

if __name__ == "__main__":
    upload_data()