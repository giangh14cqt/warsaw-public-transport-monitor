"""
Exogenous Weather Proxy Module.
Maintains backward compatibility with pipeline runners and documentation.
"""

from src.exogenous.imgw import IMGWWeatherHarvester, fetch_imgw_synoptic_archive

__all__ = ["IMGWWeatherHarvester", "fetch_imgw_synoptic_archive"]
