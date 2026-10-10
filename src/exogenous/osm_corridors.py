"""
Exogenous OSM Corridor Proxy Module.
Maintains backward compatibility with pipeline runners and documentation.
"""

from src.exogenous.osm import OSMCorridorExtractor, extract_default_corridors, DEFAULT_CORRIDORS

__all__ = ["OSMCorridorExtractor", "extract_default_corridors", "DEFAULT_CORRIDORS"]
