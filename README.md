# ML Agentic Engineer - High Stress Detection System

## Problem Statement

Build an autonomous agent pipeline that detects high-stress students and stores alerts.

## Requirements
Build an agent that meets the following requirements:
   * Reads the uploaded CSV (simulate live ingestion)

   * Detects students with stress level \> threshold (e.g., 33.3)

   * Uses lightweight ML model or heuristic rule

   * Stores flagged users in DB (`HighStressUsers` table)

## Solution
 * Since there is a dependency on a postgres database running on docker anyway, I opted to implement my solution using docker-compose.\
   So instead of using the existing lambda function templates, I created a FastAPI app that has routes to the different requirements outlined above.\
   This app is then wrapped into another docker container and the two are linked together using docker-compose.
 * I also interpreted the purpose of the model to infer a "predicted_stress_level"  using the other features in the dataset that's not "stres_level"\
   So when /process-stress-data is called, the app will train a model to predict the stress level. But then it will use "stress_level" that's already\
   given in the dataset to flag students that are high-stress and store them into the HighStressUsers table in the DB.
 * Normally for database and table management, I would use the `alembic` library to modify the tables and keep track of the changes.\
   But since this task is relatively small, I'm simply using pandas existing `to_sql` function to write to the database.


## Prerequisites

- Docker
- Docker Compose

## Testing Steps

### 1. Start the Application (in the background)

```bash
docker-compose up --build -d 
```

This will:
- Start PostgreSQL database
- Initialize the database schema
- Build and start the FastAPI application
- The API will be available at `http://localhost:8000`

### 2. Run the Agent

```bash
curl -X POST http://localhost:8000/process-stress-data -H "Content-Type: application/json" -d '{"threshold": "33.3"}'
```

This will:
- Load 1001 records from the CSV
- Train a Gradient Boosting ML model (as `predicted_stress_level`)
- Detect high-stress students (stress > 33.3)
- Store them in the database
- Return a "records updated successfully" message

### 3. Get Alerts

```bash
curl http://localhost:8000/alerts
```

### 4. Add new user
```bash
curl -X POST http://localhost:8000/user -H "Content-Type: application/json" -d '{"name": "BOOM", "id": "739e9599-ca49-40dd-8da2-0acf7506b152"}'
```

## Data flow

```
CSV Dataset -> Agent Processor (/process-stress-data) -> PostgreSQL (HighStressUsers table)

Agent Logs -> PostgreSQL (AgentLogs table)

GET /alerts endpoint -> PostgreSQL (HighStressUsers table) -> JSON Response
```

