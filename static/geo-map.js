/* Phase 16B.3 — MapLibre + MapTiler basemap. Consumes replay results only. */
(function (global) {
  "use strict";

  var STATIONS = global.TW_STATIONS || {
    VOTV: { lat: 8.4667, lon: 76.95, name: "Thiruvananthapuram" },
    VOCI: { lat: 10.15, lon: 76.4, name: "Kochi" },
    VABB: { lat: 19.1005, lon: 72.8585, name: "Mumbai" },
    VIDP: { lat: 28.5667, lon: 77.1167, name: "Delhi" },
    VECC: { lat: 22.6547, lon: 88.4467, name: "Kolkata" }
  };

  var INDIA = { center: [80.9, 22.2], zoom: 4.15, pitch: 42, bearing: -8 };
  var BOUNDS = [[66.5, 6.4], [98.4, 36.8]];

  function GeoMap() {
    this.map = null;
    this.markers = {};
    this.payload = null;
    this.focusId = null;
    this.onSelect = function () {};
    this.styleMode = "dark";
    this.cfg = null;
  }

  GeoMap.prototype.init = function (wrap, opts) {
    var self = this;
    opts = opts || {};
    this.wrap = wrap;
    this.onSelect = opts.onSelect || function () {};
    this.reduced = !!opts.reduced;
    this.tooltip = wrap.querySelector("#map-tip");
    this.callout = wrap.querySelector("#map-callout");
    this.host = wrap.querySelector("#india-gl");
    this.cfg = opts.config;
    if (!global.maplibregl || !this.cfg || !this.cfg.enabled) return false;

    if (wrap.querySelector("#india-fallback")) wrap.querySelector("#india-fallback").hidden = true;
    this.host.hidden = false;
    this.host.innerHTML = "";

    try {
      this.map = new maplibregl.Map({
        container: this.host,
        style: this.cfg.styles.dark,
        center: INDIA.center,
        zoom: INDIA.zoom,
        pitch: this.reduced ? 0 : INDIA.pitch,
        bearing: INDIA.bearing,
        maxBounds: BOUNDS,
        attributionControl: true,
        cooperativeGestures: false
      });
    } catch (e) {
      return false;
    }

    this.map.on("error", function (ev) {
      if (self._failed) return;
      var msg = String((ev && ev.error && ev.error.message) || ev.error || "");
      if (/401|403|failed to fetch|style/i.test(msg)) {
        self._failed = true;
        if (typeof opts.onFail === "function") opts.onFail();
      }
    });

    this.map.on("load", function () {
      self._terrain();
      self._markers();
      if (self.payload) self.setResults(self.payload);
    });

    wrap.querySelector("#map-zoom-in") && wrap.querySelector("#map-zoom-in").addEventListener("click", function () { self.zoom(-1); });
    wrap.querySelector("#map-zoom-out") && wrap.querySelector("#map-zoom-out").addEventListener("click", function () { self.zoom(1); });
    wrap.querySelector("#map-reset") && wrap.querySelector("#map-reset").addEventListener("click", function () { self.reset(); });
    var basemap = wrap.querySelector("#map-basemap");
    if (basemap) {
      basemap.hidden = false;
      basemap.addEventListener("click", function () { self.toggleStyle(); });
    }
    return true;
  };

  GeoMap.prototype._terrain = function () {
    if (this.reduced || !this.cfg.terrain || !this.map) return;
    try {
      if (!this.map.getSource("tw-dem")) {
        this.map.addSource("tw-dem", { type: "raster-dem", url: this.cfg.terrain, tileSize: 256 });
      }
      this.map.setTerrain({ source: "tw-dem", exaggeration: 1.15 });
    } catch (e) { /* terrain optional */ }
  };

  GeoMap.prototype._markers = function () {
    var self = this;
    Object.keys(STATIONS).forEach(function (sid) {
      var s = STATIONS[sid];
      var el = document.createElement("button");
      el.type = "button";
      el.className = "stn-marker";
      el.setAttribute("aria-label", s.name + " " + sid);
      el.innerHTML = '<span class="stn-pulse"></span><span class="stn-dot"></span><span class="stn-lab">' + sid + "</span>";
      el.addEventListener("click", function (ev) {
        ev.stopPropagation();
        self.focusStation(sid, true);
        self.onSelect(sid);
      });
      el.addEventListener("mouseenter", function (ev) { self._tip(ev, sid); });
      el.addEventListener("mouseleave", function () { if (self.tooltip) self.tooltip.hidden = true; });
      var mk = new maplibregl.Marker({ element: el, anchor: "bottom" })
        .setLngLat([s.lon, s.lat])
        .addTo(self.map);
      self.markers[sid] = { marker: mk, el: el };
    });
  };

  GeoMap.prototype._tip = function (ev, sid) {
    if (!this.tooltip) return;
    var st = STATIONS[sid];
    var rec = this._rec(sid);
    var line = "1h model probability: Awaiting replay";
    if (rec && rec.unavailable) line = "1h model probability: DATA UNAVAILABLE";
    else if (rec && rec.lead_1h_pct != null) line = "1h model probability: " + rec.lead_1h_pct + "%";
    this.tooltip.hidden = false;
    this.tooltip.style.left = "12px";
    this.tooltip.style.top = "auto";
    this.tooltip.style.bottom = "48px";
    this.tooltip.innerHTML = "<strong>" + st.name + "</strong><span>" + sid + "</span><em>" + line + "</em>";
  };

  GeoMap.prototype._rec = function (sid) {
    if (!this.payload || !this.payload.stations) return null;
    return this.payload.stations.find(function (s) { return s.station_id === sid; });
  };

  GeoMap.prototype.setResults = function (payload, opts) {
    this.payload = payload;
    var self = this;
    Object.keys(this.markers).forEach(function (sid) {
      var rec = self._rec(sid);
      var el = self.markers[sid].el;
      el.classList.remove("low", "high", "await");
      if (!rec || rec.unavailable || rec.lead_1h_probability == null) {
        el.classList.add("await");
      } else if (rec.lead_1h_probability >= 0.065) {
        el.classList.add("high");
      } else {
        el.classList.add("low");
      }
    });
    var auto = !(opts && opts.autoFocus === false);
    if (auto && payload && payload.focus_station_id) this.focusStation(payload.focus_station_id, true);
  };

  GeoMap.prototype.focusStation = function (sid, animate) {
    this.focusId = sid;
    var st = STATIONS[sid];
    if (!st || !this.map) return;
    Object.keys(this.markers).forEach(function (id) {
      this.markers[id].el.classList.toggle("sel", id === sid);
    }, this);
    var rec = this._rec(sid);
    var pct = rec && rec.lead_1h_pct != null ? rec.lead_1h_pct + "%" : "--";
    var title = (this.payload && sid === this.payload.focus_station_id) ? "HIGHEST MODEL PROBABILITY" : "STATION";
    if (this.callout) {
      this.callout.hidden = false;
      this.callout.innerHTML = "<small>" + title + "</small><strong>" + st.name + "</strong><span>" + sid + " · " + pct + "</span>";
    }
    this.map.flyTo({
      center: [st.lon, st.lat],
      zoom: Math.max(this.map.getZoom(), 6.4),
      pitch: this.reduced ? 0 : 48,
      duration: animate && !this.reduced ? 900 : 0,
      essential: true
    });
  };

  GeoMap.prototype.zoom = function (d) {
    if (!this.map) return;
    this.map.zoomTo(this.map.getZoom() - d * 0.7, { duration: this.reduced ? 0 : 250 });
  };

  GeoMap.prototype.reset = function () {
    if (!this.map) return;
    this.map.flyTo({
      center: INDIA.center, zoom: INDIA.zoom,
      pitch: this.reduced ? 0 : INDIA.pitch, bearing: INDIA.bearing,
      duration: this.reduced ? 0 : 700
    });
  };

  GeoMap.prototype.toggleStyle = function () {
    if (!this.map || !this.cfg.styles.terrain) return;
    this.styleMode = this.styleMode === "dark" ? "terrain" : "dark";
    var self = this;
    this.map.setStyle(this.cfg.styles[this.styleMode]);
    this.map.once("style.load", function () {
      self._terrain();
      Object.keys(self.markers).forEach(function (sid) {
        self.markers[sid].marker.addTo(self.map);
      });
      if (self.payload) self.setResults(self.payload);
    });
  };

  GeoMap.prototype._resize = function () {
    if (this.map) this.map.resize();
  };

  global.TWGeoMap = GeoMap;
})(window);
