"""
Exogenous Feature Ingestion: Transit Transfer Hub & Metro Proximity Metrics.
Computes:
1. interchange_line_count: Number of distinct scheduled transit lines serving each stop.
2. is_metro_proximity: Binary indicator (1/0) for stops within 300 meters of a Warsaw Metro station.
3. min_dist_to_metro_m: Physical great-circle distance in meters to the closest Metro station.
4. is_major_transfer_hub: Composite indicator for stops with >= 10 routes or adjacent to Metro.
"""

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Warsaw Metro (M1 & M2) station coordinates (lat, lon)
WARSAW_METRO_STATIONS: Dict[str, Tuple[float, float]] = {
    # Line M1 (Kabaty <-> Młociny)
    "M1_Kabaty": (52.1308, 21.0650),
    "M1_Natolin": (52.1397, 21.0569),
    "M1_Imielin": (52.1494, 21.0456),
    "M1_Stoklosy": (52.1539, 21.0347),
    "M1_Ursynow": (52.1628, 21.0283),
    "M1_Sluzew": (52.1728, 21.0267),
    "M1_Wilanowska": (52.1814, 21.0222),
    "M1_Wierzbno": (52.1903, 21.0150),
    "M1_Raclawicka": (52.1994, 21.0117),
    "M1_Pole_Mokotowskie": (52.2089, 21.0092),
    "M1_Politechnika": (52.2189, 21.0175),
    "M1_Centrum": (52.2308, 21.0108),
    "M1_Swietokrzyska": (52.2356, 21.0083),
    "M1_Ratusz_Arsenal": (52.2450, 21.0017),
    "M1_Dworzec_Gdanski": (52.2575, 20.9983),
    "M1_Plac_Wilsona": (52.2625, 20.9842),
    "M1_Marymont": (52.2683, 20.9708),
    "M1_Slodowiec": (52.2722, 20.9575),
    "M1_Stare_Bielany": (52.2783, 20.9472),
    "M1_Wawrzyszew": (52.2842, 20.9392),
    "M1_Mlociny": (52.2905, 20.9300),
    # Line M2 (Bemowo <-> Bródno)
    "M2_Bemowo": (52.2392, 20.9125),
    "M2_Ulrychow": (52.2411, 20.9286),
    "M2_Ksiecia_Janusza": (52.2400, 20.9442),
    "M2_Mlynow": (52.2378, 20.9606),
    "M2_Plocka": (52.2325, 20.9678),
    "M2_Rondo_Daszynskiego": (52.2303, 20.9839),
    "M2_Rondo_ONZ": (52.2325, 20.9989),
    "M2_Nowy_Swiat_Uniwersytet": (52.2369, 21.0183),
    "M2_Centrum_Nauki_Kopernik": (52.2403, 21.0319),
    "M2_Stadion_Narodowy": (52.2467, 21.0436),
    "M2_Dworzec_Wilenski": (52.2547, 21.0356),
    "M2_Szwedzka": (52.2617, 21.0417),
    "M2_Targowek_Mieszkaniowy": (52.2689, 21.0506),
    "M2_Trocka": (52.2742, 21.0578),
    "M2_Zacisze": (52.2856, 21.0583),
    "M2_Kondratowicza": (52.2925, 21.0431),
    "M2_Brodno": (52.2958, 21.0256),
}


class TransitInterchangeExtractor:
    """Computes route density, transfer hubs, and Metro proximity for Warsaw transit stops."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.data_dir, "transit_stop_interchanges.parquet")

    @staticmethod
    def haversine_vectorized(
        lats: np.ndarray, lons: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Compute minimum great-circle distance from each (lat, lon) to a set of target points in meters."""
        r = 6371000.0  # Earth radius in meters
        min_d = np.full(len(lats), np.inf)
        for t_lat, t_lon in targets:
            dlat = np.radians(t_lat - lats)
            dlon = np.radians(t_lon - lons)
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(np.radians(lats)) * np.cos(np.radians(t_lat)) * np.sin(dlon / 2.0) ** 2
            )
            d = 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            min_d = np.minimum(min_d, d)
        return min_d

    def extract_interchanges(
        self,
        gtfs_dir: Optional[str] = None,
        metro_threshold_meters: float = 300.0,
        output_parquet: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract interchange route count and Metro proximity for all stops in the GTFS feed.

        Parameters
        ----------
        gtfs_dir : str, optional
            Path to GTFS directory containing stops.txt, stop_times.txt, trips.txt.
        metro_threshold_meters : float
            Distance threshold in meters for metro proximity (default: 300m).
        output_parquet : str, optional
            Path to output Parquet file.
        """
        if gtfs_dir is None:
            gtfs_dirs = sorted(glob.glob("data/gtfs/202*"))
            gtfs_dir = gtfs_dirs[-1] if gtfs_dirs else "data/gtfs/2026_10_11"

        stops_file = os.path.join(gtfs_dir, "stops.txt")
        stop_times_file = os.path.join(gtfs_dir, "stop_times.txt")
        trips_file = os.path.join(gtfs_dir, "trips.txt")

        if not os.path.exists(stops_file):
            raise FileNotFoundError(f"GTFS stops file not found: {stops_file}")

        out_path = output_parquet or self.output_parquet
        con = duckdb.connect()

        try:
            has_stop_times = os.path.exists(stop_times_file) and os.path.exists(trips_file)

            if has_stop_times:
                query = f"""
                    WITH trip_routes AS (
                        SELECT CAST(trip_id AS VARCHAR) AS trip_id, CAST(route_id AS VARCHAR) AS route_id
                        FROM read_csv_auto('{trips_file}', all_varchar=true)
                    ),
                    stop_routes AS (
                        SELECT CAST(st.stop_id AS VARCHAR) AS stop_id, CAST(tr.route_id AS VARCHAR) AS route_id
                        FROM read_csv_auto('{stop_times_file}', all_varchar=true) st
                        JOIN trip_routes tr ON CAST(st.trip_id AS VARCHAR) = tr.trip_id
                        GROUP BY st.stop_id, tr.route_id
                    ),
                    route_counts AS (
                        SELECT stop_id, COUNT(DISTINCT route_id) AS interchange_line_count
                        FROM stop_routes
                        GROUP BY stop_id
                    )
                    SELECT
                        CAST(s.stop_id AS VARCHAR) AS stop_id,
                        s.stop_name,
                        CAST(s.stop_lat AS DOUBLE) AS stop_lat,
                        CAST(s.stop_lon AS DOUBLE) AS stop_lon,
                        COALESCE(rc.interchange_line_count, 1) AS interchange_line_count
                    FROM read_csv_auto('{stops_file}', all_varchar=true) s
                    LEFT JOIN route_counts rc ON CAST(s.stop_id AS VARCHAR) = rc.stop_id
                """
            else:
                query = f"""
                    SELECT
                        CAST(stop_id AS VARCHAR) AS stop_id,
                        stop_name,
                        CAST(stop_lat AS DOUBLE) AS stop_lat,
                        CAST(stop_lon AS DOUBLE) AS stop_lon,
                        1 AS interchange_line_count
                    FROM read_csv_auto('{stops_file}', all_varchar=true) s
                """

            df = con.execute(query).df()
        finally:
            con.close()

        # Compute Metro Proximity
        metro_coords = np.array(list(WARSAW_METRO_STATIONS.values()))
        min_dists = self.haversine_vectorized(
            df["stop_lat"].values, df["stop_lon"].values, metro_coords
        )
        df["min_dist_to_metro_m"] = np.round(min_dists, 1)
        df["is_metro_proximity"] = (min_dists <= metro_threshold_meters).astype(int)
        df["is_major_transfer_hub"] = (
            (df["interchange_line_count"] >= 10) | (df["is_metro_proximity"] == 1)
        ).astype(int)

        cols = [
            "stop_id",
            "stop_name",
            "interchange_line_count",
            "is_metro_proximity",
            "min_dist_to_metro_m",
            "is_major_transfer_hub",
        ]
        result_df = df[cols].copy()

        # Save to Parquet
        result_df.to_parquet(out_path, index=False, compression="snappy")
        logger.info(
            f"Extracted {len(result_df)} stop interchange records to {out_path} "
            f"({result_df['is_metro_proximity'].sum()} near Metro, "
            f"{result_df['is_major_transfer_hub'].sum()} major transfer hubs, "
            f"avg routes: {result_df['interchange_line_count'].mean():.2f})"
        )
        return result_df


def extract_default_interchanges(
    output_path: Optional[str] = None, gtfs_dir: Optional[str] = None
) -> str:
    """Convenience helper to extract and save default stop interchange metrics."""
    extractor = TransitInterchangeExtractor(
        data_dir=os.path.dirname(output_path) if output_path else "data/processed"
    )
    if output_path:
        extractor.output_parquet = output_path
    extractor.extract_interchanges(gtfs_dir=gtfs_dir, output_parquet=extractor.output_parquet)
    return extractor.output_parquet


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extractor = TransitInterchangeExtractor()
    df_out = extractor.extract_interchanges()
    print("\n=== Top Transfer Hubs by Scheduled Route Count ===")
    print(df_out.sort_values(by="interchange_line_count", ascending=False).head(15).to_string())
