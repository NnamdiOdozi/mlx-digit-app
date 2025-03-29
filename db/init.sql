-- Create the user
-- DO
-- $$
-- BEGIN
--     IF NOT EXISTS (
--         SELECT FROM pg_catalog.pg_roles
--         WHERE rolname = 'mlx_user'
--     ) THEN
--         CREATE ROLE mlx_user WITH LOGIN PASSWORD DB_PASSWORD;
--     END IF;
-- END
-- $$;

-- -- Create the database (if using a separate one)
-- CREATE DATABASE mlx_db OWNER mlx_user;

-- Connect to that database to create tables (only works manually; not inside init.sql)
-- So instead, create the table in the default DB

-- Create table (in default DB)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    prediction INTEGER NOT NULL,
    confidence NUMERIC(5, 0) NOT NULL,
    actual INTEGER NOT NULL
);