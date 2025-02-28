from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import os
from supabase import create_client, Client
load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

class ScannedIn(BaseModel):
    class_id: str

@app.post("/scanned-in", )
async def scanned_in(payload: ScannedIn):

    class_id = payload.class_id
    existing = supabase.table("scanned_items").select("*").eq("item_id", class_id).eq("cart_id", 1).execute()

    if existing.data:
        # 2a. If exists, increment quantity
        item = existing.data[0]  # Get the first row
        new_quantity = item["quantity"] + 1  # Increase the existing quantity
        updated = supabase.table("scanned_items") \
            .update({"quantity": new_quantity}) \
            .eq("item_id", class_id) \
            .eq("cart_id", 1) \
            .execute()
        return {
            "status": 201,
            "message": "Quantity updated",
            "body": updated.data
        }
    else:
        # 2b. If not found, insert a new row with quantity = 1
        inserted = supabase.table("scanned_items").insert({
            "item_id": class_id,
            "quantity": 1,
            "scanned_date": datetime.now().isoformat(),
            "cart_id": 1

        }).execute()
        return {
            "status": 201,
            "message": "New item inserted",
            "body": inserted.data
        }
@app.post("/remove-item")
async def remove_item(payload: ScannedIn):
    class_id = payload.class_id
    existing = supabase.table("scanned_items").select("*").eq("item_id", class_id).eq("cart_id", 1).execute()
    if existing.data:
        item = existing.data[0]

        new_quantity = item["quantity"] - 1
        updated = supabase.table("scanned_items") \
            .update({"quantity": new_quantity}) \
            .eq("item_id", class_id) \
            .eq("cart_id", 1) \
            .execute()
        return {
            "status": 201,
            "message": "Quantity updated",
            "body": updated.data
        }


    else:
        return {
            "status": 404,
            "message": "Item not found"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port = 8000)