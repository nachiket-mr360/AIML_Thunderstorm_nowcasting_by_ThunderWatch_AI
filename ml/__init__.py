"""Machine-learning nowcasting pipeline for the SIH 2026 thunderstorm prototype.

This package contains:

* ``train_model`` -- builds features from the Open-Meteo observation history,
  derives the surrogate convective-risk target, trains a RandomForest classifier
  and writes the model plus evaluation artifacts.
* ``predict`` -- loads the trained artifacts and scores a single current/latest
  observation row.

SCIENTIFIC DISCLAIMER (applies to every module in this package)
--------------------------------------------------------------
This prototype uses a future high-impact precipitation event as a surrogate
target because verified historical lightning/thunderstorm labels were not
available in the current dataset. It demonstrates the ML nowcasting pipeline
and should not be interpreted as an operational IMD thunderstorm or lightning
forecast.
"""

__all__ = ["train_model", "predict"]
