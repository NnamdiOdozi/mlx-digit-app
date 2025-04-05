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


-- Connect to default 'postgres' DB
\c postgres;

-- Conditionally create the database
-- DO
-- $$
-- BEGIN
--    IF NOT EXISTS (
--       SELECT FROM pg_database
--       WHERE datname = 'odozi_mlx_digit_recognizer'
--    ) THEN
CREATE DATABASE odozi_mlx_digit_recognizer;
--    END IF;
-- END
-- $$;

