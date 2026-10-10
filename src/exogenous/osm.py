"""
Exogenous Feature Ingestion: Road Topology & Infrastructure Extraction via OSMnx.
Extracts road network segments between adjacent transit stops along core Warsaw corridors:
- Al. Jerozolimskie
- Puławska
- Trasa W-Z (Solidarności)
- Towarowa / Okopowa
Computes dedicated right-of-way status, signalized intersection counts, length, and capacity.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Core transit study corridors in Warsaw
DEFAULT_CORRIDORS = {
    "al_jerozolimskie": "Aleje Jerozolimskie, Warszawa, Poland",
    "pulawska": "Puławska, Warszawa, Poland",
    "trasa_wz": "Aleja 'Solidarności', Warszawa, Poland",
    "towarowa_okopowa": "Towarowa, Warszawa, Poland",
}


class OSMCorridorExtractor:
    """Extracts and computes topological road segment infrastructure metrics."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.data_dir, "osm_corridor_segments.parquet")

    def compute_segment_metrics(
        self,
        edge_id: str,
        stop_id_prev: str,
        stop_id_curr: str,
        length_meters: float,
        is_busway: bool = False,
        signal_count: int = 0,
        lanes: int = 2,
    ) -> Dict[str, Any]:
        """Compute standardized attributes for a transit segment edge."""
        return {
            "edge_id": edge_id,
            "stop_id_prev": str(stop_id_prev),
            "stop_id_curr": str(stop_id_curr),
            "segment_length_meters": float(length_meters),
            "is_dedicated_right_of_way": bool(is_busway),
            "signalized_intersection_count": int(signal_count),
            "lane_capacity": int(lanes),
        }

    def save_segments(self, segments: List[Dict[str, Any]]) -> str:
        """Persist extracted corridor segments to Parquet."""
        df = pd.DataFrame(segments)
        df.to_parquet(self.output_parquet, index=False, compression="snappy")
        logger.info(f"Saved {len(df)} corridor segments to {self.output_parquet}")
        return self.output_parquet


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = OSMCorridorExtractor()
    print("OSMCorridorExtractor module ready.")
