CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    run_name TEXT,
    notes TEXT,
    params_json TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    episode INTEGER NOT NULL,
    score INTEGER,
    reward REAL,
    moves INTEGER,
    max_tile INTEGER,
    epsilon REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_run_episode
ON training_metrics(run_id, episode);

CREATE INDEX IF NOT EXISTS idx_training_metrics_run_phase
ON training_metrics(run_id, phase);
