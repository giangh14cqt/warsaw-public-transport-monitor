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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

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

    @staticmethod
    def haversine_distance_vectorized(
        lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
    ) -> np.ndarray:
        """Vectorized great-circle distance calculation in meters."""
        r = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    @classmethod
    def compute_shape_path_lengths(cls, gtfs_dir: str) -> pd.DataFrame:
        """
        Compute actual physical path length along GTFS shapes for each consecutive stop pair.
        Returns DataFrame with columns ['stop_id_prev', 'stop_id_curr', 'actual_path_length_meters'].
        """
        shapes_file = os.path.join(gtfs_dir, "shapes.txt")
        trips_file = os.path.join(gtfs_dir, "trips.txt")
        stop_times_file = os.path.join(gtfs_dir, "stop_times.txt")
        stops_file = os.path.join(gtfs_dir, "stops.txt")

        if not (
            os.path.exists(shapes_file)
            and os.path.exists(trips_file)
            and os.path.exists(stop_times_file)
            and os.path.exists(stops_file)
        ):
            return pd.DataFrame(
                columns=["stop_id_prev", "stop_id_curr", "actual_path_length_meters"]
            )

        con = duckdb.connect()
        try:
            shapes_df = con.execute(f"""
                SELECT shape_id, shape_pt_sequence, CAST(shape_pt_lat AS DOUBLE) as lat, CAST(shape_pt_lon AS DOUBLE) as lon
                FROM read_csv_auto('{shapes_file}')
                ORDER BY shape_id, shape_pt_sequence
            """).df()

            stops_df = con.execute(f"""
                SELECT stop_id, CAST(stop_lat AS DOUBLE) as lat, CAST(stop_lon AS DOUBLE) as lon
                FROM read_csv_auto('{stops_file}')
            """).df().set_index("stop_id")

            shape_stops_df = con.execute(f"""
                WITH trip_shapes AS (
                    SELECT trip_id, shape_id
                    FROM read_csv_auto('{trips_file}')
                    WHERE shape_id IS NOT NULL AND shape_id != ''
                ),
                st AS (
                    SELECT s.stop_id, CAST(s.stop_sequence AS INT) as seq, ts.shape_id
                    FROM read_csv_auto('{stop_times_file}') s
                    JOIN trip_shapes ts ON s.trip_id = ts.trip_id
                    GROUP BY ts.shape_id, s.stop_id, seq
                )
                SELECT shape_id, stop_id, seq
                FROM st
                ORDER BY shape_id, seq
            """).df()
        finally:
            con.close()

        # Vectorized shape cumulative distances
        shape_groups = {}
        for s_id, group in shapes_df.groupby("shape_id", sort=False):
            lats = group["lat"].values
            lons = group["lon"].values
            if len(lats) < 2:
                cum_dists = np.array([0.0])
            else:
                dists = cls.haversine_distance_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
                cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
            shape_groups[s_id] = (lats, lons, cum_dists)

        pair_lengths = []
        for s_id, group in shape_stops_df.groupby("shape_id", sort=False):
            if s_id not in shape_groups:
                continue
            sh_lats, sh_lons, cum_d = shape_groups[s_id]
            if len(sh_lats) == 0:
                continue

            stop_ids = group["stop_id"].values
            curr_idx = 0
            stop_shape_dists = []

            for sid in stop_ids:
                if sid not in stops_df.index:
                    stop_shape_dists.append(None)
                    continue
                slat = stops_df.loc[sid, "lat"]
                slon = stops_df.loc[sid, "lon"]
                if isinstance(slat, pd.Series):
                    slat, slon = slat.iloc[0], slon.iloc[0]

                sub_lats = sh_lats[curr_idx:]
                sub_lons = sh_lons[curr_idx:]
                d_sq = (sub_lats - slat) ** 2 + (sub_lons - slon) ** 2
                best_rel = int(np.argmin(d_sq))
                best_idx = curr_idx + best_rel
                curr_idx = best_idx
                stop_shape_dists.append(cum_d[best_idx])

            for i in range(len(stop_ids) - 1):
                d1 = stop_shape_dists[i]
                d2 = stop_shape_dists[i + 1]
                if d1 is not None and d2 is not None and d2 >= d1:
                    pair_lengths.append((stop_ids[i], stop_ids[i + 1], d2 - d1))

        if not pair_lengths:
            return pd.DataFrame(
                columns=["stop_id_prev", "stop_id_curr", "actual_path_length_meters"]
            )

        df_pairs = pd.DataFrame(
            pair_lengths, columns=["stop_id_prev", "stop_id_curr", "actual_path_length_meters"]
        )
        agg_pairs = (
            df_pairs.groupby(["stop_id_prev", "stop_id_curr"])["actual_path_length_meters"]
            .median()
            .reset_index()
        )
        return agg_pairs

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
        actual_path_length_meters: Optional[float] = None,
        tortuosity_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute standardized attributes for a transit segment edge."""
        act_len = float(
            round(
                actual_path_length_meters
                if actual_path_length_meters is not None
                else length_meters,
                1,
            )
        )
        tort = float(
            round(
                tortuosity_ratio
                if tortuosity_ratio is not None
                else max(1.0, act_len / max(length_meters, 1.0)),
                3,
            )
        )
        return {
            "edge_id": edge_id,
            "corridor_name": corridor_name,
            "stop_id_prev": str(stop_id_prev),
            "stop_id_curr": str(stop_id_curr),
            "stop_name_prev": str(stop_name_prev),
            "stop_name_curr": str(stop_name_curr),
            "segment_length_meters": float(round(length_meters, 1)),
            "actual_path_length_meters": act_len,
            "tortuosity_ratio": tort,
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
            signals_by_corridor[c_name] = self.load_or_fetch_traffic_signals(c_name, c_cfg["bbox"])

        # Build stop_id to corridor mapping
        stop_to_corr = dict(zip(corr_stops["stop_id"], corr_stops["corridor_name"], strict=False))

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

    def process_citywide_network(
        self,
        gtfs_dir: Optional[str] = None,
        signals_parquet: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract and compute road topology metrics for ALL directed stop-to-stop links
        across the entire Warsaw metropolitan transit network.
        """
        import glob

        if gtfs_dir is None:
            gtfs_dirs = sorted(glob.glob("data/gtfs/202*"))
            gtfs_dir = "data/gtfs/2026_10_11" if os.path.exists("data/gtfs/2026_10_11/stops.txt") else (gtfs_dirs[-1] if gtfs_dirs else "data/gtfs/2026_10_11")

        stops_file = os.path.join(gtfs_dir, "stops.txt")
        stop_times_file = os.path.join(gtfs_dir, "stop_times.txt")
        trips_file = os.path.join(gtfs_dir, "trips.txt")
        routes_file = os.path.join(gtfs_dir, "routes.txt")

        if not os.path.exists(stops_file) or not os.path.exists(stop_times_file):
            logger.warning(
                f"GTFS files not found in {gtfs_dir}, falling back to corridor filter."
            )
            return self.process_corridors(gtfs_dir=gtfs_dir)

        con = duckdb.connect()
        try:
            stop_cols = [
                c[0]
                for c in con.execute(
                    f"DESCRIBE SELECT * FROM read_csv_auto('{stops_file}') LIMIT 1"
                ).fetchall()
            ]
            street_expr = "COALESCE(street, '')" if "street" in stop_cols else "COALESCE(stop_name, '')"
            zone_expr = "zone_id" if "zone_id" in stop_cols else "''"
            stops_df = con.execute(f"""
                SELECT 
                    CAST(stop_id AS VARCHAR) as stop_id, 
                    stop_name, 
                    CAST(stop_lat AS DOUBLE) as stop_lat, 
                    CAST(stop_lon AS DOUBLE) as stop_lon,
                    {street_expr} as street,
                    {zone_expr} as zone_id
                FROM read_csv_auto('{stops_file}')
            """).df().set_index("stop_id")

            has_routes = os.path.exists(routes_file) and os.path.exists(trips_file)
            if has_routes:
                links_sql = f"""
                    WITH trips_route AS (
                        SELECT t.trip_id, r.route_id, r.route_type
                        FROM read_csv_auto('{trips_file}') t
                        JOIN read_csv_auto('{routes_file}') r ON t.route_id = r.route_id
                    ),
                    st AS (
                        SELECT s.trip_id, CAST(s.stop_sequence AS INT) as seq, s.stop_id, tr.route_id, tr.route_type
                        FROM read_csv_auto('{stop_times_file}') s
                        JOIN trips_route tr ON s.trip_id = tr.trip_id
                    ),
                    pairs AS (
                        SELECT 
                            LAG(stop_id) OVER (PARTITION BY trip_id ORDER BY seq) as stop_id_prev,
                            stop_id as stop_id_curr,
                            route_id,
                            route_type
                        FROM st
                    )
                    SELECT 
                        stop_id_prev, 
                        stop_id_curr,
                        MAX(CASE WHEN route_type = 0 OR TRY_CAST(route_id AS INT) < 100 THEN 1 ELSE 0 END) as is_tram_link
                    FROM pairs
                    WHERE stop_id_prev IS NOT NULL
                    GROUP BY stop_id_prev, stop_id_curr
                """
            else:
                links_sql = f"""
                    WITH st AS (
                        SELECT trip_id, CAST(stop_sequence AS INT) as seq, stop_id
                        FROM read_csv_auto('{stop_times_file}')
                    ),
                    pairs AS (
                        SELECT 
                            LAG(stop_id) OVER (PARTITION BY trip_id ORDER BY seq) as stop_id_prev,
                            stop_id as stop_id_curr
                        FROM st
                    )
                    SELECT 
                        stop_id_prev, 
                        stop_id_curr,
                        0 as is_tram_link
                    FROM pairs
                    WHERE stop_id_prev IS NOT NULL
                    GROUP BY stop_id_prev, stop_id_curr
                """
            links_df = con.execute(links_sql).df()
        finally:
            con.close()

        valid_mask = links_df["stop_id_prev"].isin(stops_df.index) & links_df["stop_id_curr"].isin(stops_df.index)
        links_df = links_df[valid_mask].reset_index(drop=True)

        s1_lats = stops_df.loc[links_df["stop_id_prev"], "stop_lat"].values
        s1_lons = stops_df.loc[links_df["stop_id_prev"], "stop_lon"].values
        s2_lats = stops_df.loc[links_df["stop_id_curr"], "stop_lat"].values
        s2_lons = stops_df.loc[links_df["stop_id_curr"], "stop_lon"].values

        r = 6371000.0
        dlat = np.radians(s2_lats - s1_lats)
        dlon = np.radians(s2_lons - s1_lons)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(np.radians(s1_lats)) * np.cos(np.radians(s2_lats)) * np.sin(dlon / 2.0) ** 2
        )
        dists = np.round(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)), 1)
        links_df["segment_length_meters"] = dists

        # Integrate GTFS shape path length & geometric tortuosity
        shape_pairs_df = self.compute_shape_path_lengths(gtfs_dir)
        if not shape_pairs_df.empty:
            links_df = links_df.merge(
                shape_pairs_df, on=["stop_id_prev", "stop_id_curr"], how="left"
            )
        else:
            links_df["actual_path_length_meters"] = np.nan

        links_df["actual_path_length_meters"] = np.where(
            links_df["actual_path_length_meters"].isna()
            | (links_df["actual_path_length_meters"] < links_df["segment_length_meters"]),
            links_df["segment_length_meters"],
            np.round(links_df["actual_path_length_meters"], 1),
        )
        links_df["tortuosity_ratio"] = np.round(
            links_df["actual_path_length_meters"]
            / np.maximum(links_df["segment_length_meters"], 1.0),
            3,
        )

        sig_file = signals_parquet or os.path.join(self.cache_dir, "osm_signals_warsaw_citywide.parquet")
        if not os.path.exists(sig_file):
            sig_files = glob.glob(os.path.join(self.cache_dir, "osm_signals_*.parquet"))
            if sig_files:
                sig_df = pd.concat([pd.read_parquet(f) for f in sig_files]).drop_duplicates(subset=["osmid"])
            else:
                sig_df = pd.DataFrame(columns=["osmid", "lat", "lon"])
        else:
            sig_df = pd.read_parquet(sig_file)

        sig_counts = np.zeros(len(links_df), dtype=int)
        if not sig_df.empty:
            sig_lats = sig_df["lat"].values
            sig_lons = sig_df["lon"].values
            m_per_deg_lat = 111194.9
            cos_lat = float(np.cos(np.radians(52.23)))
            m_per_deg_lon = m_per_deg_lat * cos_lat
            buffer_m = 35.0
            buffer_m_sq = buffer_m ** 2
            buffer_lat_deg = buffer_m / m_per_deg_lat
            buffer_lon_deg = buffer_m / m_per_deg_lon

            for i in range(len(links_df)):
                lat1, lon1 = s1_lats[i], s1_lons[i]
                lat2, lon2 = s2_lats[i], s2_lons[i]
                min_lat, max_lat = min(lat1, lat2) - buffer_lat_deg, max(lat1, lat2) + buffer_lat_deg
                min_lon, max_lon = min(lon1, lon2) - buffer_lon_deg, max(lon1, lon2) + buffer_lon_deg
                cand = (sig_lats >= min_lat) & (sig_lats <= max_lat) & (sig_lons >= min_lon) & (sig_lons <= max_lon)
                if not np.any(cand):
                    continue
                c_lats, c_lons = sig_lats[cand], sig_lons[cand]
                p1 = np.array([lon1 * m_per_deg_lon, lat1 * m_per_deg_lat])
                p2 = np.array([lon2 * m_per_deg_lon, lat2 * m_per_deg_lat])
                v = p2 - p1
                v_len_sq = np.dot(v, v)
                if v_len_sq == 0:
                    sig_counts[i] = len(c_lats)
                    continue
                p = np.column_stack([c_lons * m_per_deg_lon, c_lats * m_per_deg_lat])
                t = np.clip(np.dot(p - p1, v) / v_len_sq, 0.0, 1.0)
                proj = p1 + np.outer(t, v)
                dist_sq = np.sum((p - proj) ** 2, axis=1)
                sig_counts[i] = np.sum(dist_sq <= buffer_m_sq)

        links_df["signalized_intersection_count"] = sig_counts

        s1_streets = stops_df.loc[links_df["stop_id_prev"], "street"].values
        s2_streets = stops_df.loc[links_df["stop_id_curr"], "street"].values

        corridor_names = []
        dedicated_row = []
        lane_capacity = []

        for i in range(len(links_df)):
            st1 = str(s1_streets[i]).lower()
            st2 = str(s2_streets[i]).lower()
            is_tram = bool(links_df["is_tram_link"][i] == 1)

            if "jerozolimsk" in st1 or "jerozolimsk" in st2:
                c_name = "al_jerozolimskie"
                row = True
                lanes = 3
            elif "puławsk" in st1 or "puławsk" in st2:
                c_name = "pulawska"
                row = True
                lanes = 3
            elif "solidarnośc" in st1 or "solidarnośc" in st2:
                c_name = "trasa_wz"
                row = True
                lanes = 2
            elif "towarow" in st1 or "towarow" in st2 or "okopow" in st1 or "okopow" in st2:
                c_name = "towarowa_okopowa"
                row = True
                lanes = 3
            elif "łazienkowsk" in st1 or "łazienkowsk" in st2 or "stanów zjednoczonych" in st1 or "wawelsk" in st1:
                c_name = "trasa_lazienkowska"
                row = True
                lanes = 3
            elif "radzymińsk" in st1 or "radzymińsk" in st2:
                c_name = "radzyminska"
                row = True
                lanes = 3
            elif "górczewsk" in st1 or "górczewsk" in st2:
                c_name = "gorczewska"
                row = True
                lanes = 3
            elif "modlińsk" in st1 or "modlińsk" in st2:
                c_name = "modlinska"
                row = True
                lanes = 3
            elif "wisłostrad" in st1 or "czerniakowsk" in st1 or "wybrzeże" in st1:
                c_name = "wislostrada"
                row = False
                lanes = 3
            elif "kasprzak" in st1 or "prost" in st1:
                c_name = "prosta_kasprzaka"
                row = True
                lanes = 3
            elif "marszałkowsk" in st1 or "marszałkowsk" in st2:
                c_name = "marszalkowska"
                row = is_tram
                lanes = 3
            elif "jana pawła" in st1 or "chałubińsk" in st1 or "niepodległoś" in st1:
                c_name = "al_jana_pawla_ii"
                row = is_tram
                lanes = 3
            else:
                c_name = None
                row = is_tram
                lanes = 2 if dists[i] < 1500 else 3

            if is_tram:
                row = True

            corridor_names.append(c_name)
            dedicated_row.append(row)
            lane_capacity.append(lanes)

        links_df["corridor_name"] = corridor_names
        links_df["is_dedicated_right_of_way"] = dedicated_row
        links_df["lane_capacity"] = lane_capacity
        links_df["edge_id"] = links_df["stop_id_prev"].astype(str) + "_" + links_df["stop_id_curr"].astype(str)
        links_df["stop_name_prev"] = stops_df.loc[links_df["stop_id_prev"], "stop_name"].values
        links_df["stop_name_curr"] = stops_df.loc[links_df["stop_id_curr"], "stop_name"].values

        cols = [
            "edge_id",
            "corridor_name",
            "stop_id_prev",
            "stop_id_curr",
            "stop_name_prev",
            "stop_name_curr",
            "segment_length_meters",
            "actual_path_length_meters",
            "tortuosity_ratio",
            "is_dedicated_right_of_way",
            "signalized_intersection_count",
            "lane_capacity",
        ]
        result_df = links_df[cols].copy()
        if os.path.exists(self.output_parquet):
            try:
                old_df = pd.read_parquet(self.output_parquet)
                if "slug_prev" in old_df.columns and "slug_curr" in old_df.columns:
                    slug_map = old_df[["edge_id", "slug_prev", "slug_curr"]].drop_duplicates(subset=["edge_id"])
                    result_df = result_df.merge(slug_map, on="edge_id", how="left")
            except Exception:
                pass
        logger.info(
            f"Successfully processed {len(result_df)} citywide network topology segments "
            f"({result_df['is_dedicated_right_of_way'].sum()} dedicated ROW links, "
            f"avg {result_df['segment_length_meters'].mean():.1f}m, "
            f"avg {result_df['signalized_intersection_count'].mean():.2f} signals)"
        )
        return result_df

    def save_segments(self, df_or_segments: Union[pd.DataFrame, List[Dict[str, Any]]]) -> str:
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
        has_tort = False
        try:
            cols = [
                c[0]
                for c in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{self.output_parquet}')"
                ).fetchall()
            ]
            has_tort = "tortuosity_ratio" in cols
        except Exception:
            pass

        tort_sql = (
            "ROUND(AVG(tortuosity_ratio), 3) AS avg_tortuosity"
            if has_tort
            else "1.0::DOUBLE AS avg_tortuosity"
        )
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
                ROUND(AVG(lane_capacity), 2) AS avg_lane_capacity,
                {tort_sql}
            FROM read_parquet('{self.output_parquet}')
        """).fetchone()

        by_corr = (
            con.execute(f"""
            SELECT
                COALESCE(corridor_name, 'network_general') as corridor_name,
                COUNT(*) as segments,
                ROUND(SUM(segment_length_meters) / 1000.0, 2) as length_km,
                ROUND(AVG(signalized_intersection_count), 1) as avg_signals
            FROM read_parquet('{self.output_parquet}')
            GROUP BY corridor_name
            ORDER BY segments DESC
        """)
            .df()
            .to_dict(orient="records")
        )

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
            "avg_tortuosity": float(res[10]) if len(res) > 10 and res[10] is not None else 1.0,
            "corridors": by_corr,
        }


def extract_default_corridors(
    output_path: Optional[str] = None, citywide: bool = True
) -> str:
    """Convenience helper to extract and save default study corridors or citywide network."""
    extractor = OSMCorridorExtractor(
        data_dir=os.path.dirname(output_path) if output_path else "data/processed"
    )
    if output_path:
        extractor.output_parquet = output_path
    if citywide:
        df_segments = extractor.process_citywide_network()
    else:
        df_segments = extractor.process_corridors()
    if not df_segments.empty:
        extractor.save_segments(df_segments)
    return extractor.output_parquet


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extractor = OSMCorridorExtractor()
    df_segments = extractor.process_citywide_network()
    if not df_segments.empty:
        extractor.save_segments(df_segments)
        summary = extractor.get_summary()
        print("\n=== OSM Warsaw Citywide Segments Summary ===")
        print(f"Total Segments: {summary['total_segments']:,}")
        print(f"Total Corridors: {summary['total_corridors']}")
        print(f"Total Network Length: {summary['total_network_km']:,} km")
        print(f"Avg Segment Length: {summary['avg_segment_meters']} m")
        print(f"Avg Signals per Segment: {summary['avg_signals_per_segment']}")
        print(f"Total Signals Detected: {summary['total_signals_detected']:,}")
        print(f"Dedicated ROW Segments: {summary['dedicated_row_segments']:,}")
        print("\nTop Corridors:")
        for c in summary["corridors"][:10]:
            print(
                f"  - {c['corridor_name']}: {c['segments']} segments, {c['length_km']} km, avg {c['avg_signals']} signals"
            )

