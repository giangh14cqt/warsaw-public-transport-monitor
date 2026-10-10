import os
import duckdb
import pandas as pd
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = "data/delays.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trip_delays (
    feed_timestamp TIMESTAMP,
    trip_id VARCHAR,
    route_id VARCHAR,
    brigade VARCHAR,
    vehicle_id VARCHAR,
    stop_id VARCHAR,
    stop_sequence INTEGER,
    scheduled_arrival TIMESTAMP,
    rt_arrival TIMESTAMP,
    arrival_delay_seconds INTEGER,
    scheduled_departure TIMESTAMP,
    rt_departure TIMESTAMP,
    departure_delay_seconds INTEGER,
    schedule_relationship VARCHAR,
    observed_at TIMESTAMP,
    PRIMARY KEY (trip_id, stop_id, stop_sequence, observed_at)
);
"""

class DelayDatabase:
    """Manages DuckDB storage and queries for Warsaw transit delay data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self.init_schema()

    def init_schema(self) -> None:
        """Create target tables if they do not exist."""
        self.conn.execute(SCHEMA_SQL)

    def insert_delays(self, records: List[Dict[str, Any]]) -> int:
        """
        Batch insert delay records with deduplication on (trip_id, stop_id, stop_sequence, observed_at).
        Returns the number of new records inserted.
        """
        if not records:
            return 0

        df = pd.DataFrame(records)

        # Ensure datetime columns are properly formatted
        datetime_cols = [
            'feed_timestamp', 'scheduled_arrival', 'rt_arrival',
            'scheduled_departure', 'rt_departure', 'observed_at'
        ]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Insert or ignore duplicates
        initial_count = self.conn.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]

        self.conn.register("batch_df", df)
        self.conn.execute("""
            INSERT OR IGNORE INTO trip_delays
            SELECT 
                feed_timestamp,
                trip_id,
                route_id,
                brigade,
                vehicle_id,
                stop_id,
                stop_sequence,
                scheduled_arrival,
                rt_arrival,
                arrival_delay_seconds,
                scheduled_departure,
                rt_departure,
                departure_delay_seconds,
                schedule_relationship,
                observed_at
            FROM batch_df
        """)
        self.conn.unregister("batch_df")

        final_count = self.conn.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]
        return final_count - initial_count

    def get_summary_stats(self) -> Dict[str, Any]:
        """Fetch summary metrics from the delays table."""
        res = self.conn.execute("""
            SELECT 
                COUNT(*) AS total_records,
                COUNT(DISTINCT trip_id) AS active_trips,
                COUNT(DISTINCT vehicle_id) AS active_vehicles,
                COUNT(DISTINCT route_id) AS active_routes,
                MIN(arrival_delay_seconds) AS min_delay_sec,
                MAX(arrival_delay_seconds) AS max_delay_sec,
                AVG(arrival_delay_seconds) AS avg_delay_sec
            FROM trip_delays
        """).fetchdf()

        if res.empty or res['total_records'].iloc[0] == 0:
            return {
                "total_records": 0,
                "active_trips": 0,
                "active_vehicles": 0,
                "active_routes": 0,
                "min_delay_sec": None,
                "max_delay_sec": None,
                "avg_delay_sec": None,
                "db_size_bytes": self.get_storage_size_bytes()
            }

        row = res.iloc[0].to_dict()
        row["db_size_bytes"] = self.get_storage_size_bytes()
        return row

    def get_sample_rows(self, limit: int = 20, route_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve a sample of rows from the database."""
        if route_ids:
            placeholders = ",".join([f"'{r}'" for r in route_ids])
            query = f"""
                SELECT * FROM trip_delays 
                WHERE route_id IN ({placeholders})
                ORDER BY observed_at DESC, trip_id, stop_sequence 
                LIMIT {limit}
            """
        else:
            query = f"""
                SELECT * FROM trip_delays 
                ORDER BY observed_at DESC, trip_id, stop_sequence 
                LIMIT {limit}
            """
        return self.conn.execute(query).fetchdf()

    def get_storage_size_bytes(self) -> int:
        """Return the file size of the DuckDB database."""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
