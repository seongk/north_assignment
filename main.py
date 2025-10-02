"""
FastAPI Application - ML Agentic Engineer for High Stress Detection

This FastAPI application provides three main endpoints:
1. POST /user - Create a new user
2. POST /process-stress-data - Process CSV data and detect high-stress students using ML
3. GET /alerts - Retrieve high-stress alerts from the database
"""

import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

from agent import HighStressDetectionAgent

warnings.filterwarnings('ignore')

# Pydantic models
class UserCreate(BaseModel):
    id: str
    name: str

class AlertResponse(BaseModel):
    record_id: str
    timestamp: str
    mental_health_score: int

# Initialize FastAPI app
app = FastAPI()

# Database connection helper
def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "postgres"),
        database=os.environ.get("PGDATABASE", "users"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "example"),
        port=os.environ.get("PGPORT", "5432"),
    )

def get_db_engine():
    """Get SQLAlchemy engine for pandas operations"""
    connection_string = (
        f"postgresql://{os.environ.get('PGUSER', 'postgres')}:"
        f"{os.environ.get('PGPASSWORD', 'example')}@"
        f"{os.environ.get('PGHOST', 'postgres')}:"
        f"{os.environ.get('PGPORT', '5432')}/"
        f"{os.environ.get('PGDATABASE', 'users')}"
    )
    return create_engine(connection_string)


# API Routes
@app.post("/user")
async def create_user(user: UserCreate):
    """Create a new user in the database"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (id, name) VALUES (%s, %s)", (user.id, user.name))
            conn.commit()
        conn.close()

        return {"message": "New user added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-stress-data")
async def process_stress_data(stress_threshold: float = 33.3):
    """Process CSV data and detect high-stress students using ML"""
    try:
        # Connect to PostgreSQL
        engine = get_db_engine() # SQLAlchemy engine for pandas operations
        
        # Initialize the agent
        agent = HighStressDetectionAgent(engine, stress_threshold=stress_threshold)
        agent.log_action("AGENT_INITIALIZED", f"Run ID: {agent.run_id}")
        
        # Get CSV path from environment or use default
        csv_path = os.environ.get("CSV_PATH", "/app/university_mental_health_iot_dataset.csv")
        
        # Process the CSV data
        high_stress_df = agent.process_csv_data(csv_path)
        
        # Store high-stress users in database
        agent.store_high_stress_users(high_stress_df)
        
        # Save agent logsHighStressDetectionAgent
        agent.save_agent_logs()
        
        # Close database connections
        engine.dispose()
        
        agent.log_action(
            "AGENT_COMPLETE",
            f"Successfully processed records, flagged and stored high-stress users"
        )
        
        return {
            "message": "Stress data processed successfully",
            "stress_threshold": stress_threshold,
            "run_id": agent.run_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_model=List[AlertResponse])
async def get_alerts():
    """Retrieve high-stress alerts from the database"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:

            # Build query with limit
            query = f"""
                SELECT 
                    record_id,
                    timestamp,
                    mental_health_score
                FROM highstressusers
            """

            # Execute query
            cur.execute(query)
            rows = cur.fetchall()
            print(f"Found {len(rows)} high-stress alerts")

            # Format results
            alerts = []
            for row in rows:
                alert = AlertResponse(
                    record_id=row[0],
                    timestamp=row[1].strftime("%Y-%m-%dT%H:%M:%SZ") if row[2] else None,
                    mental_health_score=row[2]
                )
                alerts.append(alert)

        conn.close()
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
