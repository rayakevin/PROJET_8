CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    request_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
    client_id BIGINT,
    model_version TEXT,
    features JSONB,
    score DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    prediction INTEGER,
    decision TEXT,
    latency_ms DOUBLE PRECISION,
    preprocessing_latency_ms DOUBLE PRECISION,
    inference_latency_ms DOUBLE PRECISION,
    error_type TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_logs_request_client_status
    ON prediction_logs (request_id, client_id, status);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
    ON prediction_logs (timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_status
    ON prediction_logs (status);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_client_id
    ON prediction_logs (client_id);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_decision
    ON prediction_logs (decision);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_features_gin
    ON prediction_logs USING GIN (features);
