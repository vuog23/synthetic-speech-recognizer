"""Run with: python run.py (uses one shared set of model instances)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "3000")))
