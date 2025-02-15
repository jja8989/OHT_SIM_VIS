DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'oht_simulation') THEN
        CREATE DATABASE oht_simulation;
    END IF;
END $$;


\c oht_simulation;

CREATE TABLE IF NOT EXISTS simulation_metadata (
    simulation_id SERIAL PRIMARY KEY,
    start_time TIMESTAMP DEFAULT NOW(),
    description TEXT
);
