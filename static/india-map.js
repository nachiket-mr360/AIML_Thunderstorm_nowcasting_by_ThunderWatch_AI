/* Phase 16B.2 — 3D India station risk map (visual only). */
(function (global) {
  "use strict";

  var STATIONS = {
    VOTV: { lat: 8.4667, lon: 76.95, name: "Thiruvananthapuram" },
    VOCI: { lat: 10.15, lon: 76.4, name: "Kochi" },
    VABB: { lat: 19.1005, lon: 72.8585, name: "Mumbai" },
    VIDP: { lat: 28.5667, lon: 77.1167, name: "Delhi" },
    VECC: { lat: 22.6547, lon: 88.4467, name: "Kolkata" }
  };

  var LON0 = 82.0, LAT0 = 21.5, SCALE = 0.42;

  function project(lon, lat) {
    return { x: (lon - LON0) * SCALE, z: (LAT0 - lat) * SCALE };
  }

  function webglOk() {
    try {
      var c = document.createElement("canvas");
      return !!(c.getContext("webgl") || c.getContext("experimental-webgl"));
    } catch (e) { return false; }
  }

  function IndiaMap() {
    this.ready = false;
    this.reduced = false;
    this.onSelect = null;
    this.payload = null;
    this.focusId = null;
    this.markers = {};
  }

  IndiaMap.prototype.init = function (wrap, opts) {
    opts = opts || {};
    this.wrap = wrap;
    this.onSelect = opts.onSelect || function () {};
    this.reduced = !!opts.reduced || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.tooltip = wrap.querySelector("#map-tip");
    this.callout = wrap.querySelector("#map-callout");
    this.fallback = wrap.querySelector("#india-fallback");
    this.canvasHost = wrap.querySelector("#india-gl");

    var self = this;
    fetch("/static/india-outline.json")
      .then(function (r) { return r.json(); })
      .then(function (geo) {
        self.ring = geo.coordinates;
        if (global.THREE && webglOk()) self._initGL();
        else self._initFallback();
      })
      .catch(function () { self._initFallback(); });

    wrap.querySelector("#map-zoom-in") && wrap.querySelector("#map-zoom-in").addEventListener("click", function () { self.zoom(-0.8); });
    wrap.querySelector("#map-zoom-out") && wrap.querySelector("#map-zoom-out").addEventListener("click", function () { self.zoom(0.8); });
    wrap.querySelector("#map-reset") && wrap.querySelector("#map-reset").addEventListener("click", function () { self.reset(); });
  };

  IndiaMap.prototype._initGL = function () {
    var THREE = global.THREE;
    var host = this.canvasHost;
    host.hidden = false;
    if (this.fallback) this.fallback.hidden = true;
    var w = Math.max(host.clientWidth || 0, 480), h = Math.max(host.clientHeight || 0, 360);
    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x041018, 0.035);
    var camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 80);
    this._home = { x: 0.6, y: 11.2, z: 14.6 };
    camera.position.set(this._home.x, this._home.y, this._home.z);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0x8ec8ff, 0x061018, 0.85));
    var dir = new THREE.DirectionalLight(0xcfe8ff, 0.9);
    dir.position.set(6, 12, 4);
    scene.add(dir);
    var rim = new THREE.DirectionalLight(0x3aa0ff, 0.45);
    rim.position.set(-8, 4, -6);
    scene.add(rim);

    var shape = new THREE.Shape();
    this.ring.forEach(function (pt, i) {
      var p = project(pt[0], pt[1]);
      if (i === 0) shape.moveTo(p.x, -p.z);
      else shape.lineTo(p.x, -p.z);
    });
    var geo = new THREE.ExtrudeGeometry(shape, {
      depth: 0.55, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.06, bevelSegments: 2, steps: 1
    });
    geo.rotateX(-Math.PI / 2);
    var land = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0x163e58, roughness: 0.78, metalness: 0.12, emissive: 0x062030, emissiveIntensity: 0.35
    }));
    scene.add(land);
    var edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo, 18),
      new THREE.LineBasicMaterial({ color: 0x6ad4ff, transparent: true, opacity: 0.85 })
    );
    scene.add(edges);

    var sea = new THREE.Mesh(
      new THREE.CircleGeometry(22, 48),
      new THREE.MeshStandardMaterial({ color: 0x071422, roughness: 0.95, metalness: 0.05, transparent: true, opacity: 0.55 })
    );
    sea.rotation.x = -Math.PI / 2;
    sea.position.y = -0.08;
    scene.add(sea);

    this._addClouds(scene, THREE);
    this._addMarkers(scene, THREE);

    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.ray = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.yaw = 0.08;
    this.pitch = 0.42;
    this.dist = 18.4;
    this.targetYaw = this.yaw;
    this.targetPitch = this.pitch;
    this.targetDist = this.dist;
    this.dragging = false;
    this.last = { x: 0, y: 0 };
    this.ready = true;

    this._bind(host);
    this._loop();
    var self = this;
    window.addEventListener("resize", function () { self._resize(); });
  };

  IndiaMap.prototype._addClouds = function (scene, THREE) {
    var mat = new THREE.MeshBasicMaterial({ color: 0x8eb4d4, transparent: true, opacity: 0.07, depthWrite: false });
    this.clouds = [];
    for (var i = 0; i < 6; i++) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(2.4 + Math.random() * 1.6, 16, 12), mat.clone());
      m.position.set((Math.random() - 0.5) * 14, 1.4 + Math.random(), (Math.random() - 0.5) * 16);
      m.scale.set(1.6, 0.35, 1.1);
      scene.add(m);
      this.clouds.push(m);
    }
  };

  IndiaMap.prototype._addMarkers = function (scene, THREE) {
    var self = this;
    Object.keys(STATIONS).forEach(function (sid) {
      var s = STATIONS[sid];
      var p = project(s.lon, s.lat);
      var grp = new THREE.Group();
      grp.position.set(p.x, 0.62, p.z);
      grp.userData.sid = sid;
      var core = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x9ec8e8 })
      );
      var ring = new THREE.Mesh(
        new THREE.RingGeometry(0.18, 0.26, 24),
        new THREE.MeshBasicMaterial({ color: 0x7ad4ff, side: THREE.DoubleSide, transparent: true, opacity: 0.55 })
      );
      ring.rotation.x = -Math.PI / 2;
      grp.add(core); grp.add(ring);
      scene.add(grp);
      self.markers[sid] = { group: grp, core: core, ring: ring, base: 0.12 };
    });
  };

  IndiaMap.prototype._bind = function (host) {
    var self = this;
    var el = this.renderer.domElement;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", function (e) {
      self.dragging = true;
      self.last.x = e.clientX; self.last.y = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointerup", function (e) {
      self.dragging = false;
      self._pick(e);
    });
    el.addEventListener("pointermove", function (e) {
      if (self.dragging) {
        var dx = e.clientX - self.last.x, dy = e.clientY - self.last.y;
        self.last.x = e.clientX; self.last.y = e.clientY;
        self.targetYaw -= dx * 0.005;
        self.targetPitch = Math.max(0.22, Math.min(0.72, self.targetPitch + dy * 0.004));
        self.targetYaw = Math.max(-0.55, Math.min(0.55, self.targetYaw));
      }
      self._hover(e);
    });
    el.addEventListener("wheel", function (e) {
      e.preventDefault();
      self.zoom(e.deltaY > 0 ? 0.55 : -0.55);
    }, { passive: false });
  };

  IndiaMap.prototype.zoom = function (d) {
    this.targetDist = Math.max(9, Math.min(26, this.targetDist + d));
  };
  IndiaMap.prototype.reset = function () {
    this.targetYaw = 0.08;
    this.targetPitch = 0.42;
    this.targetDist = 18.4;
    this.focusId = null;
    if (this.callout) this.callout.hidden = true;
  };

  IndiaMap.prototype._ndc = function (e) {
    var r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    this.pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  };

  IndiaMap.prototype._hit = function (e) {
    this._ndc(e);
    this.ray.setFromCamera(this.pointer, this.camera);
    var objs = [];
    var self = this;
    Object.keys(this.markers).forEach(function (sid) { objs.push(self.markers[sid].core); });
    var hits = this.ray.intersectObjects(objs, false);
    if (!hits.length) return null;
    return hits[0].object.parent.userData.sid;
  };

  IndiaMap.prototype._hover = function (e) {
    var sid = this._hit(e);
    if (!this.tooltip) return;
    if (!sid) { this.tooltip.hidden = true; return; }
    var st = STATIONS[sid];
    var rec = this._rec(sid);
    var line = "1h model probability: Awaiting replay";
    if (rec && rec.unavailable) line = "1h model probability: DATA UNAVAILABLE";
    else if (rec && rec.lead_1h_pct != null) line = "1h model probability: " + rec.lead_1h_pct + "%";
    this.tooltip.hidden = false;
    this.tooltip.style.left = e.offsetX + 12 + "px";
    this.tooltip.style.top = e.offsetY + 12 + "px";
    this.tooltip.innerHTML = "<strong>" + st.name + "</strong><span>" + sid + "</span><em>" + line + "</em>";
  };

  IndiaMap.prototype._pick = function (e) {
    var sid = this._hit(e);
    if (sid) {
      this.focusStation(sid, true);
      this.onSelect(sid);
    }
  };

  IndiaMap.prototype._rec = function (sid) {
    if (!this.payload || !this.payload.stations) return null;
    return this.payload.stations.find(function (s) { return s.station_id === sid; });
  };

  IndiaMap.prototype.setResults = function (payload, opts) {
    this.payload = payload;
    var self = this;
    Object.keys(this.markers).forEach(function (sid) {
      var m = self.markers[sid];
      var rec = self._rec(sid);
      var p = rec && rec.lead_1h_probability != null ? rec.lead_1h_probability : null;
      var intensity = p == null ? 0 : Math.max(0, Math.min(1, p / 0.25));
      var col = p == null ? 0x9ec8e8 : (p >= 0.065 ? 0xff8a3a : 0x5ad0ff);
      m.core.material.color.setHex(col);
      m.ring.material.color.setHex(col);
      m.base = 0.12 + intensity * 0.16;
      m.core.scale.setScalar(1 + intensity * 0.9);
    });
    var auto = !(opts && opts.autoFocus === false);
    if (auto && payload && payload.focus_station_id) this.focusStation(payload.focus_station_id, true);
  };

  IndiaMap.prototype.focusStation = function (sid, animate) {
    this.focusId = sid;
    var st = STATIONS[sid];
    if (!st) return;
    var p = project(st.lon, st.lat);
    this.targetYaw = Math.max(-0.45, Math.min(0.45, -p.x * 0.04));
    this.targetDist = 13.5;
    if (this.callout) {
      var rec = this._rec(sid);
      var pct = rec && rec.lead_1h_pct != null ? rec.lead_1h_pct + "%" : "--";
      var title = (this.payload && sid === this.payload.focus_station_id) ? "HIGHEST MODEL PROBABILITY" : "STATION";
      this.callout.hidden = false;
      this.callout.innerHTML = "<small>" + title + "</small><strong>" + st.name + "</strong><span>" + sid + " · " + pct + "</span>";
    }
    if (this.reduced || !animate) return;
  };

  IndiaMap.prototype._resize = function () {
    if (!this.renderer) return;
    var w = this.canvasHost.clientWidth, h = this.canvasHost.clientHeight;
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  IndiaMap.prototype._loop = function () {
    var self = this;
    var t0 = performance.now();
    function frame(t) {
      requestAnimationFrame(frame);
      var k = self.reduced ? 1 : 0.08;
      self.yaw += (self.targetYaw - self.yaw) * k;
      self.pitch += (self.targetPitch - self.pitch) * k;
      self.dist += (self.targetDist - self.dist) * k;
      var y = Math.sin(self.pitch) * self.dist;
      var r = Math.cos(self.pitch) * self.dist;
      self.camera.position.set(Math.sin(self.yaw) * r, y, Math.cos(self.yaw) * r);
      self.camera.lookAt(0, 0.2, 0);
      if (!self.reduced && self.clouds) {
        self.clouds.forEach(function (c, i) {
          c.position.x += Math.sin(t * 0.00015 + i) * 0.004;
        });
      }
      Object.keys(self.markers).forEach(function (sid) {
        var m = self.markers[sid];
        var pulse = self.reduced ? 1 : 1 + 0.08 * Math.sin(t * 0.004 + m.base * 10);
        if (sid === self.focusId) pulse += 0.12;
        m.ring.scale.setScalar(pulse);
      });
      self.renderer.render(self.scene, self.camera);
    }
    requestAnimationFrame(frame);
  };

  IndiaMap.prototype._initFallback = function () {
    if (this.canvasHost) this.canvasHost.hidden = true;
    var svg = this.fallback;
    if (!svg) return;
    svg.hidden = false;
    var gLand = svg.querySelector("#fb-land");
    var gSt = svg.querySelector("#fb-stations");
    if (!gLand || !this.ring) return;
    var d = this.ring.map(function (pt, i) {
      var x = (pt[0] - 68) * 12.2;
      var y = (36 - pt[1]) * 12.2;
      return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }).join(" ") + " Z";
    gLand.setAttribute("d", d);
    var self = this;
    gSt.innerHTML = "";
    Object.keys(STATIONS).forEach(function (sid) {
      var s = STATIONS[sid];
      var x = (s.lon - 68) * 12.2;
      var y = (36 - s.lat) * 12.2;
      var el = document.createElementNS("http://www.w3.org/2000/svg", "g");
      el.innerHTML = '<circle cx="' + x + '" cy="' + y + '" r="6" fill="#9ec8e8"/><text x="' + (x + 10) + '" y="' + (y + 4) + '" fill="#d5e7f7" font-size="11">' + sid + "</text>";
      el.style.cursor = "pointer";
      el.addEventListener("click", function () { self.onSelect(sid); });
      gSt.appendChild(el);
    });
    this.ready = true;
    this.setResults = function (payload) {
      this.payload = payload;
    };
  };

  global.IndiaMap = IndiaMap;
  global.TW_STATIONS = STATIONS;
})(window);
