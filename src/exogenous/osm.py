"""
Exogenous Feature Ingestion: Road Topology & Infrastructure Extraction via OSMnx.
Extracts road network segments between adjacent transit stops along core Warsaw corridors:
- Al. Jerozolimskie
- Puławska
- Trasa W-Z (Solidarności)
- Towarowa / Okopowa

Computes dedicated right-of-way status, signalized intersection counts, physical segment length,
and mixed-traffic lane capacity for transport delay attribution modeling.
"""

import os
import glob
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

# Standard study corridors in Warsaw (GTFS street search pattern & spatial bounds)
DEFAULT_CORRIDORS: Dict[str, Dict[str, Any]] = {
    "al_jerozolimskie": {
        "name": "Aleje Jerozolimskie",
        "pattern": "Jerozolimsk",
        "bbox": (20.895, 52.180, 21.030, 52.235),  # (min_lon, min_lat, max_lon, max_lat)
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
    "pulawska": {
        "name": "Puławska",
        "pattern": "Puławsk",
        "bbox": (21.015, 52.095, 21.030, 52.215),
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
    "trasa_wz": {
        "name": "Trasa W-Z (Solidarności)",
        "pattern": "Solidarności",
        "bbox": (20.970, 52.235, 21.055, 52.265),
        "default_lanes": 2,
        "is_dedicated_row": True,  # Dedicated PAT (Bus+Tram) right-of-way
    },
    "towarowa_okopowa": {
        "name": "Towarowa / Okopowa",
        "pattern": "Towarow|Okopow",
        "bbox": (20.975, 52.225, 20.990, 52.255),
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
}


class OSMCorridorExtractor:
    """Extracts, maps, and calculates road & transit infrastructure topology metrics."""

    def __init__(self, data_dir: str = "data/processed", cache_dir: str = "data/cache"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.data_dir, "osm_corridor_segments.parquet")

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in meters."""
        r = 6371000.0  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

    def extract_corridor_stops(
        self,
        gtfs_dir: str,
        zone_id: int = 1,
        corridors: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Filter GTFS stops located on the specified corridors within the study zone."""
        stops_file = os.path.join(gtfs_dir, "stops.txt")
        if not os.path.exists(stops_file):
            raise FileNotFoundError(f"GTFS stops file not found: {stops_file}")

        stops = pd.read_csv(stops_file, dtype={"stop_id": str, "stop_code": str})
        stops["stop_id"] = stops["stop_id"].astype(str)

        if "zone_id" in stops.columns and zone_id is not None:
            stops = stops[stops["zone_id"] == zone_id].copy()

        configs = corridors or DEFAULT_CORRIDORS
        matched_dfs = []
        for corr_id, cfg in configs.items():
            pattern = cfg["pattern"]
            mask = stops["street"].str.contains(pattern, case=False, na=False)
            corr_df = stops[mask].copy()
            corr_df["corridor_name"] = corr_id
            matched_dfs.append(corr_df)

        if not matched_dfs:
            return pd.DataFrame()

        result = pd.concat(matched_dfs, ignore_index=True)
        logger.info(f"Identified {len(result)} GTFS stops across {len(configs)} corridors")
        return result

    def extract_adjacent_stop_pairs(
        self,
        gtfs_dir: str,
        corridor_stops: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract directed consecutive stop pairs (stop_id_prev, stop_id_curr)
        traversed by scheduled trips along the target corridors.
        """
        if corridor_stops is None:
            corridor_stops = self.extract_corridor_stops(gtfs_dir)

        if corridor_stops.empty:
            return pd.DataFrame()

        stop_times_file = os.path.join(gtfs_dir, "stop_times.txt")
        if not os.path.exists(stop_times_file):
            raise FileNotFoundError(f"GTFS stop_times file not found: {stop_times_file}")

        target_stops = set(corridor_stops["stop_id"].unique())
        st = pd.read_csv(
            stop_times_file,
            usecols=["trip_id", "stop_id", "stop_sequence"],
            dtype={"trip_id": str, "stop_id": str, "stop_sequence": int},
        )

        # Filter to trips passing through any target corridor stop
        target_trips = st[st["stop_id"].isin(target_stops)]["trip_id"].unique()
        st_sub = st[st["trip_id"].isin(target_trips)].sort_values(by=["trip_id", "stop_sequence"])

        # Shift to determine next sequential stop along the trip
        st_sub["next_stop"] = st_sub.groupby("trip_id")["stop_id"].shift(-1)
        st_sub = st_sub.dropna(subset=["next_stop"])

        # Retain consecutive pairs where both stops reside on corridor
        corridor_pairs = st_sub[
            (st_sub["stop_id"].isin(target_stops)) & (st_sub["next_stop"].isin(target_stops))
        ][["stop_id", "next_stop"]].drop_duplicates()

        corridor_pairs.rename(
            columns={"stop_id": "stop_id_prev", "next_stop": "stop_id_curr"}, inplace=True
        )

        logger.info(f"Extracted {len(corridor_pairs)} unique directed corridor stop pairs")
        return corridor_pairs

    def load_or_fetch_traffic_signals(
        self,
        corridor_name: str,
        bbox: Tuple[float, float, float, float],
    ) -> pd.DataFrame:
        """
        Load cached traffic signals or query Overpass API via OSMnx.
        bbox format: (min_lon, min_lat, max_lon, max_lat)
        """
        cache_path = os.path.join(self.cache_dir, f"osm_signals_{corridor_name}.parquet")
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {len(df)} cached signals for {corridor_name}")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}")

        # Attempt OSMnx Overpass query
        try:
            import osmnx as ox

            ox.settings.use_cache = True
            ox.settings.cache_folder = os.path.join(self.cache_dir, "osmnx")
            logger.info(f"Querying OSMnx traffic signals for {corridor_name} (bbox: {bbox})...")
            gdf = ox.features_from_bbox(bbox=bbox, tags={"highway": "traffic_signals"})

            pts = []
            for idx, row in gdf.iterrows():
                if hasattr(row, "geometry") and row.geometry.geom_type == "Point":
                    osmid = idx[1] if isinstance(idx, tuple) else idx
                    pts.append({"osmid": str(osmid), "lat": row.geometry.y, "lon": row.geometry.x})

            df = pd.DataFrame(pts)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached {len(df)} traffic signals to {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"OSMnx Overpass query unavailable for {corridor_name}: {e}")
            return pd.DataFrame(columns=["osmid", "lat", "lon"])

    def count_signals_along_segment(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        signals_df: pd.DataFrame,
        buffer_degrees: float = 0.00035,  # ~35 meters
    ) -> int:
        """Count traffic signal points within a corridor buffer between two stops."""
        if signals_df.empty:
            return 0

        min_lat = min(lat1, lat2) - buffer_degrees
        max_lat = max(lat1, lat2) + buffer_degrees
        min_lon = min(lon1, lon2) - buffer_degrees
        max_lon = max(lon1, lon2) + buffer_degrees

        sub = signals_df[
            (signals_df["lat"] >= min_lat)
            & (signals_df["lat"] <= max_lat)
            & (signals_df["lon"] >= min_lon)
            & (signals_df["lon"] <= max_lon)
        ]
        if sub.empty:
            return 0

        # Point-to-segment distance check
        p1 = np.array([lon1, lat1])
        p2 = np.array([lon2, lat2])
        v = p2 - p1
        v_len_sq = np.dot(v, v)

        if v_len_sq == 0:
            return len(sub)

        signals_in_buffer = 0
        for _, s in sub.iterrows():
            p = np.array([s["lon"], s["lat"]])
            # Project point p onto segment [p1, p2]
            t = np.clip(np.dot(p - p1, v) / v_len_sq, 0.0, 1.0)
            proj = p1 + t * v
            dist_sq = np.sum((p - proj) ** 2)
            if dist_sq <= (buffer_degrees**2):
                signals_in_buffer += 1

        return signals_in_buffer

    def compute_segment_metrics(
        self,
        edge_id: str,
        stop_id_prev: str,
        stop_id_curr: str,
        stop_name_prev: str,
        stop_name_curr: str,
        corridor_name: str,
        length_meters: float,
        is_dedicated_right_of_way: bool = True,
        signal_count: int = 0,
        lane_capacity: int = 3,
    ) -> Dict[str, Any]:
        """Compute standardized attributes for a transit segment edge."""
        return {
            "edge_id": edge_id,
            "corridor_name": corridor_name,
            "stop_id_prev": str(stop_id_prev),
            "stop_id_curr": str(stop_id_curr),
            "stop_name_prev": str(stop_name_prev),
            "stop_name_curr": str(stop_name_curr),
            "segment_length_meters": float(round(length_meters, 1)),
            "is_dedicated_right_of_way": bool(is_dedicated_right_of_way),
            "signalized_intersection_count": int(signal_count),
            "lane_capacity": int(lane_capacity),
        }

    def process_corridors(
        self,
        gtfs_dir: str = "data/gtfs/2026_10_04",
        corridors: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Extract and compute road topology metrics for all target corridors."""
        configs = corridors or DEFAULT_CORRIDORS
        corr_stops = self.extract_corridor_stops(gtfs_dir, zone_id=1, corridors=configs)
        if corr_stops.empty:
            logger.warning("No corridor stops found.")
            return pd.DataFrame()

        stops_indexed = corr_stops.set_index("stop_id")
        pairs = self.extract_adjacent_stop_pairs(gtfs_dir, corridor_stops=corr_stops)
        if pairs.empty:
            logger.warning("No adjacent stop pairs found.")
            return pd.DataFrame()

        # Load cached or fetched traffic signals for all corridors
        signals_by_corridor: Dict[str, pd.DataFrame] = {}
        for c_name, c_cfg in configs.items():
            signals_by_corridor[c_name] = self.load_or_fetch_traffic_signals(
                c_name, c_cfg["bbox"]
            )

        # Build stop_id to corridor mapping
        stop_to_corr = dict(zip(corr_stops["stop_id"], corr_stops["corridor_name"]))

        segments: List[Dict[str, Any]] = []
        for _, row in pairs.iterrows():
            s1 = row["stop_id_prev"]
            s2 = row["stop_id_curr"]
            if s1 not in stops_indexed.index or s2 not in stops_indexed.index:
                continue

            row1 = stops_indexed.loc[s1]
            row2 = stops_indexed.loc[s2]
            # Handle duplicates if multiple platforms share stop_id
            if isinstance(row1, pd.DataFrame):
                row1 = row1.iloc[0]
            if isinstance(row2, pd.DataFrame):
                row2 = row2.iloc[0]

            lat1, lon1 = float(row1["stop_lat"]), float(row1["stop_lon"])
            lat2, lon2 = float(row2["stop_lat"]), float(row2["stop_lon"])
            dist = self.haversine_distance(lat1, lon1, lat2, lon2)

            corr_name = stop_to_corr.get(s1, stop_to_corr.get(s2, "unknown"))
            cfg = configs.get(corr_name, {})

            # Traffic signals
            sig_df = signals_by_corridor.get(corr_name, pd.DataFrame())
            sig_count = self.count_signals_along_segment(lat1, lon1, lat2, lon2, sig_df)

            edge_metrics = self.compute_segment_metrics(
                edge_id=f"{s1}_{s2}",
                stop_id_prev=s1,
                stop_id_curr=s2,
                stop_name_prev=str(row1["stop_name"]),
                stop_name_curr=str(row2["stop_name"]),
                corridor_name=corr_name,
                length_meters=dist,
                is_dedicated_right_of_way=cfg.get("is_dedicated_row", True),
                signal_count=sig_count,
                lane_capacity=cfg.get("default_lanes", 3),
            )
            segments.append(edge_metrics)

        df = pd.DataFrame(segments)
        logger.info(f"Successfully processed {len(df)} corridor topology segments")
        return df

    def save_segments(
        self, df_or_segments: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> str:
        """Persist extracted corridor segments to Parquet."""
        if isinstance(df_or_segments, pd.DataFrame):
            df = df_or_segments
        else:
            df = pd.DataFrame(df_or_segments)

        df.to_parquet(self.output_parquet, index=False, compression="snappy")
        logger.info(f"Saved {len(df)} corridor segments to {self.output_parquet}")
        return self.output_parquet

    def get_summary(self) -> Dict[str, Any]:
        """Fetch descriptive summary of the stored corridor segments via DuckDB."""
        if not os.path.exists(self.output_parquet):
            return {"status": "empty", "records": 0}

        con = duckdb.connect(":memory:")
        res = con.execute(f"""
            SELECT 
                COUNT(*) AS total_segments,
                COUNT(DISTINCT corridor_name) AS total_corridors,
                ROUND(SUM(segment_length_meters) / 1000.0, 2) AS total_network_km,
                ROUND(AVG(segment_length_meters), 1) AS avg_segment_meters,
                ROUND(MIN(segment_length_meters), 1) AS min_segment_meters,
                ROUND(MAX(segment_length_meters), 1) AS max_segment_meters,
                ROUND(AVG(signalized_intersection_count), 2) AS avg_signals_per_segment,
                SUM(signalized_intersection_count) AS total_signals_detected,
                SUM(CASE WHEN is_dedicated_right_of_way THEN 1 ELSE 0 END) AS dedicated_row_segments,
                ROUND(AVG(lane_capacity), 2) AS avg_lane_capacity
            FROM read_parquet('{self.output_parquet}')
        """).fetchone()

        by_corr = con.execute(f"""
            SELECT 
                corridor_name, 
                COUNT(*) as segments,
                ROUND(SUM(segment_length_meters) / 1000.0, 2) as length_km,
                ROUND(AVG(signalized_intersection_count), 1) as avg_signals
            FROM read_parquet('{self.output_parquet}')
            GROUP BY corridor_name
            ORDER BY segments DESC
        """).df().to_dict(orient="records")

        return {
            "total_segments": int(res[0]),
            "total_corridors": int(res[1]),
            "total_network_km": float(res[2]),
            "avg_segment_meters": float(res[3]),
            "min_segment_meters": float(res[4]),
            "max_segment_meters": float(res[5]),
            "avg_signals_per_segment": float(res[6]),
            "total_signals_detected": int(res[7]),
            "dedicated_row_segments": int(res[8]),
            "avg_lane_capacity": float(res[9]),
            "corridors": by_corr,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extractor = OSMCorridorExtractor()
    df_segments = extractor.process_corridors()
    if not df_segments.empty:
        extractor.save_segments(df_segments)
        summary = extractor.get_summary()
        print("\n=== OSM Corridor Segments Summary ===")
        print(f"Total Segments: {summary['total_segments']}")
        print(f"Total Corridors: {summary['total_corridors']}")
        print(f"Total Network Length: {summary['total_network_km']} km")
        print(f"Avg Segment Length: {summary['avg_segment_meters']} m")
        print(f"Avg Signals per Segment: {summary['avg_signals_per_segment']}")
        print(f"Total Signals: {summary['total_signals_detected']}")
        print(f"Dedicated ROW Segments: {summary['dedicated_row_segments']}")
        print("\nCorridor Breakdown:")
        for c in summary["corridors"]:
            print(f"  - {c['corridor_name']}: {c['segments']} segments, {c['length_km']} km, avg {c['avg_signals']} signals")
