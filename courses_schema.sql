-- Course Recommender — Postgres Schema & Seed Data
-- Run: psql -d courses_db -f schema.sql

CREATE TABLE IF NOT EXISTS courses (
    id                     SERIAL PRIMARY KEY,
    title                  TEXT,
    url                    TEXT,
    price                  NUMERIC(10, 2),
    number_of_subscribers  INTEGER,
    number_of_reviews      INTEGER,
    num_lectures           INTEGER,
    rating                 NUMERIC(4, 2),
    content_duration       NUMERIC(6, 1),
    duration_hours         INTEGER,
    published_timestamp    TIMESTAMPTZ,
    skills                 TEXT,          -- comma-separated, e.g. 'python,fastapi,docker'
    description            TEXT,
    level                  TEXT,
    instructor             TEXT
);

-- GIN index on the skills expression enables fast && (overlap) queries
CREATE INDEX IF NOT EXISTS idx_courses_skills
    ON courses USING GIN (string_to_array(skills, ','));