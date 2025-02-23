import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/oht_simulation")
