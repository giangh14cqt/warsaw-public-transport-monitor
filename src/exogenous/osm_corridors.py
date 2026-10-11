"""
Exogenous OSM Corridor Proxy Module.
Maintains backward compatibility with pipeline runners and documentation.
"""

from src.exogenous.osm import DEFAULT_CORRIDORS, OSMCorridorExtractor, extract_default_corridors

__all__ = ["OSMCorridorExtractor", "extract_default_corridors", "DEFAULT_CORRIDORS"]
