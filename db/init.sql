CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL
);

-- Table for storing high-stress student alerts
CREATE TABLE IF NOT EXISTS HighStressUsers (
  id SERIAL PRIMARY KEY,
  record_id VARCHAR(255) NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  location_id INTEGER,
  temperature_celsius DECIMAL(5,2),
  humidity_percent DECIMAL(5,2),
  air_quality_index INTEGER,
  noise_level_db DECIMAL(5,2),
  lighting_lux DECIMAL(7,2),
  crowd_density INTEGER,
  stress_level INTEGER NOT NULL,
  sleep_hours DECIMAL(4,2),
  mood_score DECIMAL(3,1),
  mental_health_status INTEGER,
  UNIQUE(record_id)
);

-- Table for logging agent actions
CREATE TABLE IF NOT EXISTS AgentLogs (
  id SERIAL PRIMARY KEY,
  run_id VARCHAR(255) NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  action VARCHAR(255) NOT NULL,
  details TEXT,
  records_processed INTEGER,
  records_flagged INTEGER,
  status VARCHAR(50)
);
