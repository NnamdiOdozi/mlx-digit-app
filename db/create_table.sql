-- Connect to the newly created (or existing) database
\connect odozi_mlx_digit_recognizer;

-- Create table (in default DB)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    prediction INTEGER NOT NULL,
    confidence NUMERIC(5, 0) NOT NULL,
    actual INTEGER NOT NULL
);