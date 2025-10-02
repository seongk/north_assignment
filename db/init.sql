CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL
);

-- Table for storing high-stress student alerts
CREATE TABLE IF NOT EXISTS HighStressUsers (
  id SERIAL PRIMARY KEY,
  record_id VARCHAR(255) NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  mental_health_score INTEGER,
  UNIQUE(record_id)
);

-- Table for logging agent actions
CREATE TABLE IF NOT EXISTS AgentLogs (
  id SERIAL PRIMARY KEY,
  run_id VARCHAR(255) NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  action VARCHAR(255) NOT NULL,
  details TEXT,
  status VARCHAR(50)
);
