from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Test vector functionality
test_data = {
    'id': 'test1',
    'content': 'This is a test document',
    'embedding': [0.1] * 1536  # Create a simple test embedding
}

# Insert test data
try:
    result = supabase.table('documents').insert(test_data).execute()
    print("Successfully inserted test document with vector embedding!")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Query to verify table structure
try:
    result = supabase.table('documents').select("*").limit(1).execute()
    print("\nTable structure verification:")
    print(result)
except Exception as e:
    print(f"Error querying table: {e}")
