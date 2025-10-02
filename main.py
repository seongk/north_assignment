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
from datetime import datetime
import uuid
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
warnings.filterwarnings('ignore')

# Pydantic models
class UserCreate(BaseModel):
    id: str
    name: str

class AlertResponse(BaseModel):
    record_id: str
    timestamp: str
    stress_level: int
    sleep_hours: float
    mood_score: float
    location_id: str

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

class HighStressDetectionAgent:
    """
    Autonomous agent for detecting and flagging high-stress students using ML
    """

    def __init__(self, db_engine, stress_threshold=33.3):
        self.engine = db_engine      # SQLAlchemy engine for pandas operations
        self.run_id = str(uuid.uuid4())
        self.stress_threshold = stress_threshold
        self.actions_log = []
        self.model = None
        self.scaler = None
        # remove stress_level so we can predict predicted_stress_level without it as a feature
        self.feature_columns = [
            'temperature_celsius', 'humidity_percent', 'air_quality_index',
            'noise_level_db', 'lighting_lux', 'crowd_density',
            'sleep_hours', 'mood_score'
        ]
        self.model_metrics = {}
        
    def log_action(self, action, details="", status="success"):
        """Log agent actions for transparency and debugging"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details,
            "status": status
        }
        self.actions_log.append(log_entry)
        print(f"[AGENT LOG] {action}: {details}")
        
    def train_ml_model(self, df):
        """
        Train a machine learning regression model to predict stress level
        """
        self.log_action("ML_TRAINING_START", "Training regression model on dataset")

        # Prepare features and target
        X = df[self.feature_columns].copy()

        # Target score value is stress_level
        y = df['stress_level']

        # Handle any missing values
        X = X.fillna(X.mean())

        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features for better model performance
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Gradient Boosting Regressor
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred_test = self.model.predict(X_test_scaled)

        # Calculate metrics
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        self.model_metrics = {
            'test_r2': round(test_r2, 4),
            'test_rmse': round(test_rmse, 4),
            'test_mae': round(test_mae, 4),
        }

        self.log_action(
            "ML_TRAINING_COMPLETE",
            f"Model trained - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}"
        )

        return self.model, self.model_metrics


    def predict_stress_level(self, row):
        """Use trained ML model to predict mental health score"""

        # Prepare features
        features = np.array([[
            row['temperature_celsius'],
            row['humidity_percent'],
            row['air_quality_index'],
            row['noise_level_db'],
            row['lighting_lux'],
            row['crowd_density'],
            row['sleep_hours'],
            row['mood_score']
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        score = self.model.predict(features_scaled)[0]

        return round(score, 2)

    def process_csv_data(self, csv_path):
        """Read and process the CSV dataset to identify high-stress students using ML"""
        self.log_action("START_PROCESSING", f"Reading CSV from {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)
        self.log_action("CSV_LOADED", f"Loaded {len(df)} records from dataset")

        # Train ML model on the entire dataset
        self.train_ml_model(df)

        # Filter for high-stress records (stress_level > 33.3)
        high_stress_df = df[df['stress_level'] > self.stress_threshold].copy()

        # Use ML model to predict mental health scores for high-stress records
        self.log_action("ML_PREDICTION_START", "Predicting mental health scores using trained model")

        high_stress_df['predicted_stress_level'] = high_stress_df.apply(
            self.predict_stress_level, axis=1
        )

        # Generate unique record IDs
        high_stress_df['record_id'] = high_stress_df.apply(
            lambda row: f"intake-{row.name:06d}", axis=1
        )

        self.log_action(
            "ML_PREDICTION_COMPLETE",
            f"Predicted mental health scores for {len(high_stress_df)} records using ML model"
        )

        return high_stress_df


    def store_high_stress_users(self, high_stress_df):
        """Store flagged high-stress users in the database"""
        self.log_action("START_DB_STORAGE", f"Storing {len(high_stress_df)} records")
        
        # Create a copy to avoid modifying the original DataFrame
        df_to_store = high_stress_df.copy()

        # Convert timestamp string to datetime if needed
        if 'timestamp' in df_to_store.columns:
            df_to_store['timestamp'] = pd.to_datetime(df_to_store['timestamp'])

        # Map predicted_stress_level to mental_health_score for database compatibility
        if 'predicted_stress_level' in df_to_store.columns:
            df_to_store['mental_health_score'] = df_to_store['predicted_stress_level']
            df_to_store = df_to_store.drop(columns=['predicted_stress_level'])

        # Write to database using SQLAlchemy engine
        df_to_store.to_sql(
            "highstressusers",
            self.engine,
            if_exists="append",
            index=False,
            method='multi'
        )

        self.log_action(
            "DB_STORAGE_COMPLETE", f"Successfully stored {len(high_stress_df)} new records"
        )

        return len(high_stress_df)


    def save_agent_logs(self):
        """Save logs from agent"""
        df = pd.DataFrame(self.actions_log)
        # Add run_id and counts to each log entry
        df['run_id'] = self.run_id

        df.to_sql(
            "agentlogs",
            self.engine,
            if_exists="append",
            index=False
        )

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
                    mental_health_score,
                    timestamp,
                    stress_level,
                    sleep_hours,
                    mood_score,
                    location_id
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
                    timestamp=row[2].strftime("%Y-%m-%dT%H:%M:%SZ") if row[2] else None,
                    stress_level=row[3],
                    sleep_hours=float(row[4]) if row[4] else 0,
                    mood_score=float(row[5]) if row[5] else 0,
                    location_id=str(row[6]) if row[6] is not None else "",
                )
                alerts.append(alert)

        conn.close()
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
