-- Course Recommender — Postgres Schema & Seed Data
-- Run: psql -d courses_db -f schema.sql

CREATE TABLE IF NOT EXISTS courses (
    id              SERIAL PRIMARY KEY,
    title           TEXT         NOT NULL,
    description     TEXT         NOT NULL,
    skills          TEXT[]       NOT NULL DEFAULT '{}',
    rating          NUMERIC(3,2) NOT NULL CHECK (rating BETWEEN 0 AND 5),
    level           TEXT         NOT NULL CHECK (level IN ('Beginner','Intermediate','Advanced')),
    duration_hours  INT          NOT NULL,
    instructor      TEXT         NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- GIN index enables fast && (overlap) queries on the skills array
CREATE INDEX IF NOT EXISTS idx_courses_skills ON courses USING GIN (skills);

-- Seed data
INSERT INTO courses (title, description, skills, rating, level, duration_hours, instructor) VALUES
('Python for Data Science',      'Master NumPy, Pandas and data analysis in Python.',          ARRAY['python','numpy','pandas','data analysis'],                    4.8, 'Beginner',     12, 'Dr. Sarah Chen'),
('FastAPI in Production',        'Build and deploy async REST APIs with FastAPI + Postgres.',  ARRAY['python','fastapi','postgresql','rest api','docker'],          4.9, 'Intermediate', 18, 'Marco Ricci'),
('Machine Learning Fundamentals','Regression to neural networks with scikit-learn & TF.',     ARRAY['python','machine learning','scikit-learn','tensorflow'],       4.7, 'Intermediate', 24, 'Dr. Aisha Patel'),
('PostgreSQL Advanced Techniques','Indexes, window functions, JSONB and query tuning.',        ARRAY['postgresql','sql','database design','performance tuning'],     4.6, 'Advanced',     15, 'Jan Kowalski'),
('Deep Learning with PyTorch',   'CNNs, RNNs, transformers and model deployment.',            ARRAY['python','pytorch','deep learning','machine learning'],         4.9, 'Advanced',     30, 'Dr. Sarah Chen'),
('SQL for Analysts',             'Complex queries, aggregations and reporting pipelines.',     ARRAY['sql','postgresql','data analysis','business intelligence'],    4.5, 'Beginner',     10, 'Emma Rodriguez'),
('Full-Stack Python',            'FastAPI backend + React frontend deployed on Railway.',      ARRAY['python','fastapi','react','postgresql','docker'],              4.8, 'Intermediate', 28, 'Alex Kim'),
('Data Engineering with Spark',  'Batch and streaming pipelines with PySpark and Kafka.',     ARRAY['python','spark','kafka','data engineering','sql'],             4.7, 'Advanced',     26, 'Ravi Sharma');
