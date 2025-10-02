import uuid
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HighStressDetectionAgent:
    """
    Autonomous agent for detecting and flagging high-stress students using ML
    """

    def __init__(self, db_engine, stress_threshold: float = 33.3):
        self.engine = db_engine  # SQLAlchemy engine for pandas operations
        self.run_id = str(uuid.uuid4())
        self.stress_threshold = stress_threshold
        self.actions_log: List[dict] = []
        self.model: GradientBoostingRegressor | None = None
        self.scaler: StandardScaler | None = None
        # remove stress_level so we can predict predicted_stress_level without it as a feature
        self.feature_columns = [
            'temperature_celsius', 'humidity_percent', 'air_quality_index',
            'noise_level_db', 'lighting_lux', 'crowd_density',
            'sleep_hours', 'mood_score'
        ]
        self.model_metrics: dict = {}

    def log_action(self, action: str, details: str = "", status: str = "success") -> None:
        """Log agent actions for transparency and debugging"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details,
            "status": status,
        }
        self.actions_log.append(log_entry)
        print(f"[AGENT LOG] {action}: {details}")

    def train_ml_model(self, df: pd.DataFrame):
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
            verbose=0,
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
            f"Model trained - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}",
        )

        return self.model, self.model_metrics

    def predict_stress_level(self, row: pd.Series) -> float:
        """Use trained ML model to predict mental health score"""
        features = np.array([[
            row['temperature_celsius'],
            row['humidity_percent'],
            row['air_quality_index'],
            row['noise_level_db'],
            row['lighting_lux'],
            row['crowd_density'],
            row['sleep_hours'],
            row['mood_score'],
        ]])
        features_scaled = self.scaler.transform(features)
        score = self.model.predict(features_scaled)[0]
        return round(score, 2)

    def process_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Read and process the CSV dataset to identify high-stress students using ML"""
        self.log_action("START_PROCESSING", f"Reading CSV from {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)
        self.log_action("CSV_LOADED", f"Loaded {len(df)} records from dataset")

        # Train ML model on the entire dataset
        self.train_ml_model(df)

        # Filter for high-stress records (stress_level > threshold)
        high_stress_df = df[df['stress_level'] > self.stress_threshold].copy()

        # Predict stress for filtered records
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
            f"Predicted mental health scores for {len(high_stress_df)} records using ML model",
        )

        return high_stress_df

    def store_high_stress_users(self, high_stress_df: pd.DataFrame) -> int:
        """Store flagged high-stress users in the database"""
        self.log_action("START_DB_STORAGE", f"Storing {len(high_stress_df)} records")

        # Prepare dataframe for DB write
        df_to_store = high_stress_df.copy()
        if 'timestamp' in df_to_store.columns:
            df_to_store['timestamp'] = pd.to_datetime(df_to_store['timestamp'])
        if 'predicted_stress_level' in df_to_store.columns:
            df_to_store['mental_health_score'] = df_to_store['predicted_stress_level']
            df_to_store = df_to_store.drop(columns=['predicted_stress_level'])

        # Write via SQLAlchemy engine
        df_to_store.to_sql(
            "highstressusers",
            self.engine,
            if_exists="append",
            index=False,
            method='multi',
        )

        self.log_action(
            "DB_STORAGE_COMPLETE",
            f"Successfully stored {len(high_stress_df)} new records",
        )
        return len(high_stress_df)

    def save_agent_logs(self) -> None:
        """Save logs from agent"""
        df = pd.DataFrame(self.actions_log)
        df['run_id'] = self.run_id
        df.to_sql(
            "agentlogs",
            self.engine,
            if_exists="append",
            index=False,
        )


