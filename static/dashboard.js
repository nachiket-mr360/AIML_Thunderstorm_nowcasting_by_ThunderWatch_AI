/* ==========================================================================
   SIH 2026 / SIH26072 — Phase 9 decision-support dashboard.
   Vanilla JS, no build step.

   Primary source of truth:
     GET /api/prediction   Phase 7/8 1-hour thunderstorm nowcast
   Supporting:
     GET /api/history      recent atmospheric hours for charts
     GET /api/health       optional status
     GET /api/evaluation   Phase 1 proxy metrics (reference only)
     GET /api/scenario     Phase 1 historical proxy demo

   The browser never invents weather, probability, or surrounding spatial risk.
   Probability is a model score — never labelled "confidence".
   ========================================================================== */

(function () {
  "use strict";

  var CONDITION_FIELDS = [
    { key: "temperature_2m", label: "Temperature" },
    { key: "relative_humidity_2m", label: "Relative Humidity" },
    { key: "surface_pressure", label: "Surface Pressure" },
    { key: "wind_speed_10m", label: "Wind Speed" },
    { key: "wind_direction_10m", label: "Wind Direction" },
    { key: "precipitation", label: "Precipitation" },
    { key: "cloud_cover", label: "Cloud Cover" }
  ];

  var METRIC_FIELDS = [
    { key: "accuracy", label: "Accuracy", digits: 4 },
    { key: "precision", label: "Precision", digits: 4 },
    { key: "recall", label: "Recall", digits: 4 },
    { key: "f1_score", label: "F1-score", digits: 4 },
    { key: "roc_auc", label: "ROC-AUC", digits: 4 }
  ];

  var TOP_FEATURES = 15;
  var COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  var EMPTY = "\u2014";
  var LOCKED_THRESHOLD = 0.0775;
  var EXPECTED_LEAD_HOURS = 1;

  /* Map state: a FeatureGroup so multi-point predictions can be added later
     without inventing values today. Only real API points are drawn. */
  var mapState = {
    map: null,
    riskLayer: null,
    baseReady: false,
    defaultLat: null,
    defaultLon: null,
    locationName: ""
  };

  var chartState = { temperature: null, precipitation: null };
  var scenarioRun = false;

  /* ------------------------------ helpers -------------------------------- */

  function el(id) { return document.getElementById(id); }

  function setText(id, value) {
    var node = el(id);
    if (node) { node.textContent = value; }
  }

  function isNum(value) {
    return typeof value === "number" && isFinite(value);
  }

  function oneDecimal(value) {
    return isNum(value) ? value.toFixed(1) : null;
  }

  function compassPoint(degrees) {
    if (!isNum(degrees)) { return ""; }
    var index = Math.round((((degrees % 360) + 360) % 360) / 22.5) % 16;
    return COMPASS[index];
  }

  function formatUtc(iso) {
    if (!iso) { return EMPTY; }
    var date = new Date(iso);
    if (isNaN(date.getTime())) { return String(iso); }
    return date.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, "Z").replace("Z", " UTC");
  }

  function formatHourTick(iso) {
    var date = new Date(iso);
    if (isNaN(date.getTime())) { return iso; }
    return String(date.getUTCHours()).padStart(2, "0") + ":00";
  }

  function fmtCoord(value, digits) {
    return isNum(value) ? Number(value).toFixed(digits) : EMPTY;
  }

  /* --------------------------------- API --------------------------------- */

  var Api = {
    get: function (path) {
      return fetch(path, { headers: { Accept: "application/json" }, cache: "no-store" })
        .then(function (response) {
          return response.json()
            .catch(function () { return null; })
            .then(function (data) {
              return { ok: response.ok, status: response.status, data: data };
            });
        })
        .catch(function (error) {
          return { ok: false, status: 0, data: null, networkError: String(error) };
        });
    }
  };

  /* ------------------------- operational states -------------------------- */

  var STATUS_COPY = {
    pending: "Loading",
    live: "LIVE \u00b7 latest available data",
    api_unavailable: "API unavailable",
    prediction_unavailable: "Prediction unavailable",
    stale: "Data stale / unavailable"
  };

  function setStatus(state) {
    var pill = el("live-status");
    var text = el("live-status-text");
    if (!pill || !text) { return; }
    var klass = {
      pending: "status-pending",
      live: "status-live",
      api_unavailable: "status-error",
      prediction_unavailable: "status-error",
      stale: "status-warn"
    }[state] || "status-pending";
    pill.className = "status-pill " + klass;
    pill.setAttribute("data-state", state);
    text.textContent = STATUS_COPY[state] || state;
  }

  function showError(message, detail) {
    var banner = el("error-banner");
    if (!banner) { return; }
    setText("error-title", message);
    setText("error-detail", detail || "");
    var detailNode = el("error-detail");
    if (detailNode) { detailNode.hidden = !detail; }
    banner.hidden = false;
  }

  function hideError() {
    var banner = el("error-banner");
    if (banner) { banner.hidden = true; }
  }

  function describeFailure(result) {
    if (result.networkError || result.status === 0) {
      return {
        state: "api_unavailable",
        message: "API unavailable. No nowcast was generated.",
        detail: "Could not reach the prediction API" +
          (result.networkError ? ": " + result.networkError : ".")
      };
    }
    var data = result.data || {};
    if (data.error || !result.ok) {
      var code = data.code || ("http_" + result.status);
      var stale = /stale/i.test(String(code)) || /stale/i.test(String(data.message || ""));
      return {
        state: stale ? "stale" : "prediction_unavailable",
        message: stale
          ? "Atmospheric data is stale or unavailable. No nowcast was generated."
          : "Prediction unavailable. No nowcast was generated.",
        detail: "Backend reported " + code + ": " + (data.message || "no detail")
      };
    }
    return {
      state: "prediction_unavailable",
      message: "Prediction unavailable. No nowcast was generated.",
      detail: "Unexpected response (HTTP " + result.status + ")."
    };
  }

  /* ------------------------------ clear live ----------------------------- */

  function clearLiveValues() {
    var label = el("risk-label");
    if (label) {
      label.textContent = EMPTY;
      label.className = "risk-label risk-label-empty";
    }
    var card = el("risk-section");
    if (card) { card.className = "card risk-card hero"; }

    [
      "probability", "lead-time", "threshold", "feature-timestamp",
      "prediction-timestamp", "model-identifier", "data-source",
      "target-timestamp", "data-age", "feature-completeness"
    ].forEach(function (id) { setText(id, EMPTY); });

    var bar = el("probability-bar");
    if (bar) { bar.style.width = "0%"; }

    var grid = el("conditions-grid");
    if (grid) { grid.innerHTML = ""; }
    setText("conditions-note", "Awaiting live data\u2026");
    setText("skill-note",
      "Measured Phase 6 test skill for this locked threshold is shown after a live prediction loads. " +
      "An alert is a screening signal, not a definite thunderstorm forecast.");

    clearRiskMarkers();
    setText("map-mode-note", "Point / location-based risk");
    setText("spatial-capability", "Single validated point (not a forecast grid)");
    setThresholdMarker(LOCKED_THRESHOLD);
  }

  /* Hero threshold tick: marks the locked decision threshold on the probability
     bar. Position comes from the API value; nothing here is invented. */
  function setThresholdMarker(thresholdValue) {
    var mark = el("prob-threshold-mark");
    if (mark && isNum(thresholdValue)) {
      mark.style.left = Math.max(0, Math.min(100, thresholdValue * 100)).toFixed(3) + "%";
      mark.title = "Locked decision threshold " + thresholdValue.toFixed(4);
    }
    setText("prob-threshold-mark-label",
      isNum(thresholdValue) ? thresholdValue.toFixed(4) : EMPTY);
  }

  /* ------------------------------ risk card ------------------------------ */

  function renderRisk(data) {
    var elevated = data.predicted_class === 1;
    var label = el("risk-label");
    if (label) {
      label.textContent = data.risk_label || EMPTY;
      label.className = "risk-label " + (elevated ? "risk-label-elevated" : "risk-label-low");
    }
    var card = el("risk-section");
    if (card) {
      card.className = "card risk-card hero " + (elevated ? "is-elevated" : "is-low");
    }

    setText("probability",
      isNum(data.probability) ? (data.probability * 100).toFixed(2) + "%" : EMPTY);
    var bar = el("probability-bar");
    if (bar && isNum(data.probability)) {
      bar.style.width = Math.max(0, Math.min(100, data.probability * 100)).toFixed(2) + "%";
      bar.className = "prob-fill " + (elevated ? "prob-alert" : "prob-low");
    }

    var lead = data.lead_time_hours;
    setText("lead-time",
      lead === EXPECTED_LEAD_HOURS || lead === "1"
        ? "1 hour"
        : (isNum(lead) ? lead + " hour" + (lead === 1 ? "" : "s") : EMPTY));

    var thresholdValue = isNum(data.threshold)
      ? Number(data.threshold)
      : LOCKED_THRESHOLD;
    setText("threshold", thresholdValue.toFixed(4));
    setThresholdMarker(thresholdValue);

    setText("feature-timestamp", formatUtc(data.feature_timestamp || data.observation_timestamp_utc));
    setText("prediction-timestamp", formatUtc(data.prediction_timestamp));
    setText("target-timestamp", formatUtc(data.target_timestamp));

    var modelId = data.model_identifier || (data.model && data.model.name) || EMPTY;
    setText("model-identifier", modelId);

    var provenance = ((data.source || {}).provenance) || {};
    var sourceBits = [];
    if (provenance.provider) { sourceBits.push(provenance.provider); }
    if (provenance.endpoint) {
      try {
        sourceBits.push(new URL(provenance.endpoint).hostname);
      } catch (err) {
        sourceBits.push(String(provenance.endpoint));
      }
    }
    if ((data.source || {}).frame_origin) {
      sourceBits.push(data.source.frame_origin);
    }
    setText("data-source", sourceBits.length ? sourceBits.join(" · ") : EMPTY);

    var age = (data.temporal_semantics || {}).data_age_hours;
    setText("data-age", isNum(age) ? age.toFixed(2) + " h" : EMPTY);

    var completeness = data.feature_completeness || {};
    if (isNum(completeness.n_features_provided) && isNum(completeness.n_features_expected)) {
      setText("feature-completeness",
        completeness.n_features_provided + "/" + completeness.n_features_expected +
        (completeness.all_features_finite ? " finite" : ""));
    } else {
      setText("feature-completeness", EMPTY);
    }

    var skill = data.measured_skill_reference || {};
    if (isNum(skill.recall) || isNum(skill.precision)) {
      setText("skill-note",
        "Phase 6 measured test skill at threshold " +
        (isNum(data.threshold) ? Number(data.threshold).toFixed(4) : "0.0775") +
        ": recall " + (isNum(skill.recall) ? skill.recall.toFixed(3) : EMPTY) +
        ", precision " + (isNum(skill.precision) ? skill.precision.toFixed(3) : EMPTY) +
        ". An alert is a screening signal, not a definite thunderstorm. " +
        "Probability is a model score, not confidence.");
    }
  }

  /* --------------------------- atmosphere cards -------------------------- */

  function renderConditions(data) {
    var grid = el("conditions-grid");
    if (!grid) { return; }
    grid.innerHTML = "";

    var weather = data.current_weather || data.input_atmospheric_conditions || {};
    CONDITION_FIELDS.forEach(function (field) {
      var entry = weather[field.key] || {};
      var wrapper = document.createElement("div");
      wrapper.className = "condition";
      wrapper.setAttribute("data-var", field.key);

      var name = document.createElement("span");
      name.className = "condition-name";
      name.textContent = field.label;

      var value = document.createElement("span");
      value.className = "condition-value";
      var formatted = oneDecimal(entry.value);
      value.textContent = formatted === null ? EMPTY : formatted;

      if (formatted !== null) {
        var unit = document.createElement("span");
        unit.className = "condition-unit";
        unit.textContent = (entry.unit || "") +
          (field.key === "wind_direction_10m" ? " " + compassPoint(entry.value) : "");
        value.appendChild(unit);
      }

      wrapper.appendChild(name);
      wrapper.appendChild(value);
      grid.appendChild(wrapper);
    });

    var featureTs = data.feature_timestamp || data.observation_timestamp_utc;
    setText("conditions-note", "Feature hour " + formatUtc(featureTs));
  }

  /* --------------------------------- map --------------------------------- */

  function clearRiskMarkers() {
    if (mapState.riskLayer) {
      mapState.riskLayer.clearLayers();
    }
  }

  function riskIcon(elevated) {
    var color = elevated ? "#a3520a" : "#14713f";
    return L.divIcon({
      className: "risk-marker-wrap",
      html: '<span class="risk-marker" style="background:' + color +
            ';border-color:' + color + '"></span>',
      iconSize: [18, 18],
      iconAnchor: [9, 9]
    });
  }

  /**
   * Render real prediction points only.
   * Contract: each entry must come from the API (or the known served cell for
   * that same live prediction). Never synthesize neighbouring cell risks.
   */
  function renderSpatialRisk(points) {
    if (!mapState.map || !mapState.riskLayer) { return; }
    clearRiskMarkers();

    var list = Array.isArray(points) ? points.slice() : [];
    list.forEach(function (point) {
      if (!isNum(point.latitude) || !isNum(point.longitude)) { return; }
      var elevated = point.predicted_class === 1;
      var marker = L.marker([point.latitude, point.longitude], {
        icon: riskIcon(elevated),
        keyboard: true,
        title: point.risk_label || "Point risk"
      });

      var html =
        "<div class='map-popup'>" +
        "<strong>" + (point.risk_label || "Point / location-based risk") + "</strong><br>" +
        "Probability: " + (isNum(point.probability)
          ? (point.probability * 100).toFixed(2) + "%"
          : EMPTY) + "<br>" +
        "Threshold: " + (isNum(point.threshold)
          ? Number(point.threshold).toFixed(4)
          : EMPTY) + "<br>" +
        "Lead time: " + (point.lead_time_hours != null
          ? point.lead_time_hours + " h"
          : EMPTY) + "<br>" +
        "<span class='mono'>" + fmtCoord(point.latitude, 6) + "&deg; N, " +
        fmtCoord(point.longitude, 5) + "&deg; E</span><br>" +
        "<em>Single grid cell &mdash; not a spatial forecast grid</em>" +
        "</div>";
      marker.bindPopup(html);
      mapState.riskLayer.addLayer(marker);
    });

    if (list.length === 1) {
      mapState.map.setView([list[0].latitude, list[0].longitude], 11, { animate: false });
    } else if (list.length > 1) {
      mapState.map.fitBounds(mapState.riskLayer.getBounds().pad(0.35), { animate: false });
    }

    setText("map-mode-note",
      list.length <= 1
        ? "Point / location-based risk"
        : ("Multi-point risk (" + list.length + " validated cells)"));
    setText("spatial-capability",
      list.length <= 1
        ? "Single validated point (not a forecast grid)"
        : (list.length + " validated prediction points (no invented cells)"));
  }

  function updateMapFromPrediction(data) {
    var served = data.served_grid_cell ||
      (((data.source || {}).provenance) || {}).served_grid_cell || {};
    var lat = isNum(served.latitude) ? served.latitude : mapState.defaultLat;
    var lon = isNum(served.longitude) ? served.longitude : mapState.defaultLon;

    if (isNum(lat) && isNum(lon)) {
      setText("served-coords",
        fmtCoord(lat, 6) + "\u00b0 N, " + fmtCoord(lon, 5) + "\u00b0 E");
    }

    // Exactly one real prediction point — never fabricate neighbours.
    renderSpatialRisk([{
      latitude: lat,
      longitude: lon,
      probability: data.probability,
      predicted_class: data.predicted_class,
      risk_label: data.risk_label,
      threshold: data.threshold,
      lead_time_hours: data.lead_time_hours,
      model_identifier: data.model_identifier || ((data.model || {}).name)
    }]);
  }

  function initMap() {
    var container = el("map");
    if (!container) { return; }

    mapState.defaultLat = parseFloat(container.dataset.latitude);
    mapState.defaultLon = parseFloat(container.dataset.longitude);
    mapState.locationName = container.dataset.location || "Thiruvananthapuram";

    if (typeof L === "undefined") {
      container.innerHTML =
        '<div class="map-fallback">Map library unavailable. Served grid: ' +
        mapState.defaultLat + "\u00b0 N, " + mapState.defaultLon + "\u00b0 E " +
        "(point / location-based risk only).</div>";
      return;
    }

    var map = L.map(container, {
      center: [mapState.defaultLat, mapState.defaultLon],
      zoom: 11,
      scrollWheelZoom: false,
      attributionControl: true
    });

    // Esri World Street Map basemap. Key-free: the ArcGIS REST tile endpoint
    // serves these tiles without any API key, token or referer allow-list.
    // NOTE the Esri path order is {z}/{y}/{x}, not {z}/{x}/{y}.
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 19,
        attribution:
          'Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, ' +
          'iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), ' +
          'TomTom, 2012'
      }
    ).addTo(map);

    // Empty risk layer ready for 1..N real prediction points.
    mapState.riskLayer = L.featureGroup().addTo(map);
    mapState.map = map;
    mapState.baseReady = true;

    // Context crosshair only — explicitly NOT a prediction footprint / km grid.
    L.circleMarker([mapState.defaultLat, mapState.defaultLon], {
      radius: 5,
      color: "#0b5c96",
      weight: 1,
      fillColor: "#1273b8",
      fillOpacity: 0.25,
      interactive: false
    }).addTo(map);

    setTimeout(function () { map.invalidateSize(); }, 80);
  }

  /* -------------------------------- charts ------------------------------- */

  function chartLibraryAvailable() {
    return typeof Chart !== "undefined";
  }

  function noteTrend(message) {
    var note = el("trend-note");
    if (note) {
      note.textContent = message;
      note.hidden = false;
    }
  }

  function clearTrendNote() {
    var note = el("trend-note");
    if (note) { note.hidden = true; }
  }

  function renderTrends(data) {
    if (!chartLibraryAvailable()) {
      noteTrend("Chart library could not be loaded. Live atmospheric values remain available above.");
      return;
    }
    var hours = (data && data.hours) || [];
    if (!hours.length) {
      noteTrend("No recent hourly atmospheric data was returned, so no trend is plotted.");
      return;
    }
    clearTrendNote();

    var labels = hours.map(function (hour) { return formatHourTick(hour.timestamp_utc); });
    var temperatures = hours.map(function (h) { return isNum(h.temperature_2m) ? h.temperature_2m : null; });
    var humidities = hours.map(function (h) { return isNum(h.relative_humidity_2m) ? h.relative_humidity_2m : null; });
    var precipitation = hours.map(function (h) { return isNum(h.precipitation) ? h.precipitation : null; });
    var units = data.units || {};

    setText("trend-window",
      hours.length + " hours · " + formatUtc(data.window_start_utc) +
      " → " + formatUtc(data.window_end_utc));

    if (chartState.temperature) { chartState.temperature.destroy(); }
    if (chartState.precipitation) { chartState.precipitation.destroy(); }

    chartState.temperature = new Chart(el("trend-chart"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Temperature (" + (units.temperature_2m || "\u00b0C") + ")",
            data: temperatures,
            borderColor: "#c9453c",
            backgroundColor: "rgba(201, 69, 60, 0.10)",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.25,
            spanGaps: false,
            yAxisID: "y"
          },
          {
            label: "Relative Humidity (" + (units.relative_humidity_2m || "%") + ")",
            data: humidities,
            borderColor: "#0b5c96",
            backgroundColor: "rgba(11, 92, 150, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.25,
            spanGaps: false,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { boxWidth: 10, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              title: function (items) {
                return formatUtc(hours[items[0].dataIndex].timestamp_utc);
              }
            }
          }
        },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } },
               grid: { display: false } },
          y: { position: "left", title: { display: true, text: "\u00b0C", font: { size: 10 } },
               ticks: { font: { size: 10 } } },
          y1: { position: "right", suggestedMin: 0, suggestedMax: 100,
                title: { display: true, text: "%", font: { size: 10 } },
                ticks: { font: { size: 10 } }, grid: { display: false } }
        }
      }
    });

    chartState.precipitation = new Chart(el("precip-chart"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Precipitation (" + (units.precipitation || "mm") + ")",
          data: precipitation,
          backgroundColor: "#2f8f5f",
          borderWidth: 0,
          barPercentage: 0.9,
          categoryPercentage: 0.95
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { boxWidth: 10, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              title: function (items) {
                return formatUtc(hours[items[0].dataIndex].timestamp_utc);
              }
            }
          }
        },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } },
               grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: "mm", font: { size: 10 } },
               ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  /* -------------------- Phase 1 reference (demoted) ---------------------- */

  function renderEvaluation(data) {
    setText("evaluation-title", data.section_title || el("evaluation-title").textContent);
    setText("metrics-note", data.caveat || "");

    var metrics = data.metrics || {};
    var grid = el("metrics-grid");
    if (grid) {
      grid.innerHTML = "";
      METRIC_FIELDS.forEach(function (field) {
        var card = document.createElement("div");
        card.className = "metric-card";
        var name = document.createElement("span");
        name.className = "metric-name";
        name.textContent = field.label;
        var value = document.createElement("span");
        value.className = "metric-value-lg";
        value.textContent = isNum(metrics[field.key])
          ? metrics[field.key].toFixed(field.digits)
          : EMPTY;
        card.appendChild(name);
        card.appendChild(value);
        grid.appendChild(card);
      });
    }

    var sources = data.sources || {};
    setText("metrics-source",
      "Phase 1 surrogate metrics from " +
      (sources.evaluation || "outputs/evaluation.json") +
      ". Not thunderstorm detection accuracy.");

    var image = el("confusion-image");
    var matrix = data.confusion_matrix || {};
    if (image && matrix.image_url) { image.src = matrix.image_url; }

    var decoded = el("matrix-decode");
    if (decoded && Array.isArray(matrix.matrix) && matrix.matrix.length === 2) {
      var labels = matrix.labels || ["Low Risk", "Elevated Risk"];
      var tn = matrix.matrix[0][0], fp = matrix.matrix[0][1];
      var fn = matrix.matrix[1][0], tp = matrix.matrix[1][1];
      decoded.textContent =
        "Surrogate test decode: " + tn + " TN, " + fp + " FP, " + fn + " FN, " + tp + " TP (" +
        labels[0] + " / " + labels[1] + ").";
    }
  }

  function renderImportance(data) {
    var list = el("importance-list");
    if (!list) { return; }
    var rows = (data.feature_importance || []).slice();
    if (!rows.length) {
      list.innerHTML = "";
      setText("importance-source", "No feature importance values were returned.");
      return;
    }
    rows.sort(function (a, b) { return b.importance - a.importance; });
    var top = rows.slice(0, TOP_FEATURES);
    var max = top[0].importance || 1;
    list.innerHTML = "";
    top.forEach(function (row) {
      var wrapper = document.createElement("div");
      wrapper.className = "importance-row";
      var name = document.createElement("span");
      name.className = "importance-name";
      name.textContent = row.feature;
      name.title = row.feature;
      var track = document.createElement("div");
      track.className = "importance-track";
      var fill = document.createElement("div");
      fill.className = "importance-fill";
      fill.style.width = Math.max(0, (row.importance / max) * 100).toFixed(1) + "%";
      track.appendChild(fill);
      var value = document.createElement("span");
      value.className = "importance-value";
      value.textContent = isNum(row.importance) ? row.importance.toFixed(4) : EMPTY;
      wrapper.appendChild(name);
      wrapper.appendChild(track);
      wrapper.appendChild(value);
      list.appendChild(wrapper);
    });
    setText("importance-source",
      "Top " + top.length + " Phase 1 proxy predictors (not the thunderstorm nowcast feature set).");
  }

  /* ------------------------- historical scenario ------------------------- */

  function setScenarioBusy(busy) {
    var button = el("scenario-button");
    if (!button) { return; }
    button.disabled = busy;
    button.textContent = busy
      ? "Running\u2026"
      : (scenarioRun ? "Run Historical Scenario Again" : "Run Historical Scenario");
  }

  function showScenarioError(message) {
    var box = el("scenario-error");
    if (!box) { return; }
    box.textContent = message;
    box.hidden = false;
    var result = el("scenario-result");
    if (result) { result.hidden = true; }
  }

  function hideScenarioError() {
    var box = el("scenario-error");
    if (box) { box.hidden = true; }
  }

  function fillTable(id, rows) {
    var body = el(id);
    if (!body) { return; }
    body.innerHTML = "";
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var label = document.createElement("td");
      label.textContent = row.label;
      var value = document.createElement("td");
      value.textContent = row.value;
      tr.appendChild(label);
      tr.appendChild(value);
      body.appendChild(tr);
    });
  }

  function humaniseFeature(name) {
    return name.replace(/_/g, " ");
  }

  function renderScenario(data) {
    scenarioRun = true;
    hideScenarioError();
    setText("scenario-timestamp", formatUtc(data.historical_timestamp));

    var elevated = data.predicted_class === 1;
    var labelNode = el("scenario-label");
    if (labelNode) {
      labelNode.textContent = data.risk_label || EMPTY;
      labelNode.className = "scenario-label " +
        (elevated ? "scenario-label-elevated" : "scenario-label-low");
    }

    setText("scenario-probability",
      isNum(data.probability) ? (data.probability * 100).toFixed(2) + "%" : EMPTY);
    var bar = el("scenario-probability-bar");
    if (bar && isNum(data.probability)) {
      bar.style.width = Math.max(0, Math.min(100, data.probability * 100)).toFixed(2) + "%";
    }

    var weather = data.input_atmospheric_conditions || {};
    var inputRows = CONDITION_FIELDS.map(function (field) {
      var entry = weather[field.key] || {};
      var value = entry.value;
      return {
        label: field.label,
        value: isNum(value)
          ? value.toFixed(1) + (entry.unit ? " " + entry.unit : "")
          : EMPTY
      };
    });
    fillTable("scenario-inputs", inputRows);

    var lags = data.historical_lag_and_rolling_features || {};
    var lagRows = Object.keys(lags).map(function (key) {
      return {
        label: humaniseFeature(key),
        value: isNum(lags[key]) ? Number(lags[key]).toFixed(4) : String(lags[key])
      };
    });
    fillTable("scenario-lags", lagRows);

    var outcome = data.actual_proxy_outcome || {};
    setText("scenario-future-precip",
      isNum(outcome.future_precip_3h)
        ? outcome.future_precip_3h.toFixed(2) + " mm"
        : EMPTY);
    setText("scenario-actual-label", outcome.actual_label || EMPTY);
    setText("scenario-outcome-note", outcome.note || "");
    setText("scenario-leak-strip",
      data.future_values_used_for_prediction === false
        ? "No future values were used as model inputs for this historical prediction."
        : "Check leakage fields carefully.");
    setText("scenario-explanation", data.explanation || "");
    setText("scenario-caveat", data.scenario_caveat || data.caveat || "");

    var result = el("scenario-result");
    if (result) { result.hidden = false; }
  }

  function runScenario() {
    setScenarioBusy(true);
    hideScenarioError();
    return Api.get("/api/scenario")
      .then(function (result) {
        if (result.ok && result.data && !result.data.error) {
          renderScenario(result.data);
        } else {
          showScenarioError("The historical scenario could not be loaded. " +
            describeFailure(result).detail);
        }
      })
      .catch(function (error) {
        showScenarioError("The historical scenario could not be loaded. " + String(error));
      })
      .then(function () { setScenarioBusy(false); });
  }

  /* ------------------------------- loaders ------------------------------- */

  function setBusy(busy) {
    var button = el("refresh-button");
    if (!button) { return; }
    button.disabled = busy;
    button.textContent = busy ? "Refreshing\u2026" : "Refresh Nowcast";
  }

  function loadStatic() {
    return Api.get("/api/evaluation").then(function (result) {
      if (result.ok && result.data && !result.data.error) {
        renderEvaluation(result.data);
        renderImportance(result.data);
      } else {
        var failure = describeFailure(result);
        setText("metrics-source", "Model evaluation could not be loaded. " + failure.detail);
        setText("importance-source", "Feature importance could not be loaded. " + failure.detail);
      }
    });
  }

  function loadLive() {
    setBusy(true);
    setStatus("pending");
    hideError();
    clearLiveValues();

    return Api.get("/api/prediction")
      .then(function (prediction) {
        if (prediction.ok && prediction.data && !prediction.data.error) {
          renderRisk(prediction.data);
          renderConditions(prediction.data);
          updateMapFromPrediction(prediction.data);
          setStatus("live");
          window.__PHASE9_LAST_PREDICTION__ = prediction.data;
        } else {
          var failure = describeFailure(prediction);
          setStatus(failure.state);
          showError(failure.message, failure.detail);
          window.__PHASE9_LAST_PREDICTION__ = null;
        }
      })
      .then(function () {
        return Api.get("/api/history");
      })
      .then(function (history) {
        if (history.ok && history.data && !history.data.error) {
          renderTrends(history.data);
        } else {
          noteTrend("Recent hourly atmospheric data could not be loaded. " +
            describeFailure(history).detail);
        }
      })
      .then(function () { setBusy(false); })
      .catch(function (error) {
        setBusy(false);
        setStatus("api_unavailable");
        showError("API unavailable. No nowcast was generated.", String(error));
        clearLiveValues();
      });
  }

  /* --------------------------------- init -------------------------------- */

  function init() {
    initMap();
    setStatus("pending");

    var button = el("refresh-button");
    if (button) {
      button.addEventListener("click", function () { loadLive(); });
    }

    var scenarioButton = el("scenario-button");
    if (scenarioButton) {
      scenarioButton.addEventListener("click", function () { runScenario(); });
    }

    window.addEventListener("resize", function () {
      if (mapState.map) { mapState.map.invalidateSize(); }
    });

    loadStatic();
    loadLive();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
