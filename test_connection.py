import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Load your keys from the .env file
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# 2. Try to connect
try:
    pc = Pinecone(api_key=api_key)
    index = pc.Index("mediquery")
    stats = index.describe_index_stats()
    print("✅ SUCCESS: Connected to Pinecone!")
    print(f"📊 Your Index Stats: {stats}")
except Exception as e:
    print(f"❌ ERROR: Could not connect. Check your API key in the .env file.")
    print(f"Details: {e}")