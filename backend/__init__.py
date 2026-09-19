"""Phase 8 -- backend package for the SIH 2026 thunderstorm nowcast API.

SIH26072 -- "AIML based nowcasting of thunderstorm and lightning".

This package holds the service layer that the Flask application in ``app.py``
exposes over HTTP. It contains no model mathematics, no feature engineering and
no prediction logic of its own: everything scientific is delegated to the
already-verified Phase 7 engine, ``ml/predict_thunderstorm_nowcast.py``.
"""
