(function () {
  "use strict";

  var lastPayload = null;
  var lastSatellite = {};
  var focusId = null;
  var appMode = "replay";
  var liveByStation = {};
  var liveFetchToken = 0;
  var liveRunning = false;
  var liveAllDone = false;
  var liveSortLead = "1h";
  var liveMap = null;
  var liveMapPending = null;
  var motion = "full";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var LIVE_ORDER = ["VABB", "VIDP", "VECC", "VOCI", "VOTV"];
  var SIGNAL_LABELS = [
    ["temperature_2m", "Temperature", "°C"],
    ["relative_humidity_2m", "Humidity", "%"],
    ["surface_pressure", "Pressure", "hPa"],
    ["wind_speed_10m", "Wind", "km/h"],
    ["precipitation", "Precipitation", "mm"],
    ["cloud_cover", "Cloud Cover", "%"],
    ["nwp_cape", "CAPE", "J/kg"],
    ["nwp_convective_inhibition", "CIN", "J/kg"],
    ["nwp_lifted_index", "Lifted Index", "K"]
  ];

  function $(id) { return document.getElementById(id); }
  function toLocalInput(iso) { return String(iso).replace("Z", "").slice(0, 16); }
  function isoFromLocal(raw) {
    raw = String(raw || "").trim();
    if (!raw) {
      var def = document.body && document.body.getAttribute("data-default-ts");
      return def || "";
    }
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(raw)) raw += ":00";
    if (raw.slice(-1) !== "Z" && raw.indexOf("+") === -1 && raw.indexOf("-", 11) === -1) raw += "Z";
    return raw;
  }

  /* ---------------- atmosphere canvas ---------------- */
  var canvas = $("wx-canvas");
  var ctx = canvas && canvas.getContext("2d");
  var rain = [];
  var lastFlash = 0;
  var flash = 0;

  function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    rain = [];
    var n = motion === "off" || reduced ? 0 : motion === "reduced" ? 40 : 110;
    for (var i = 0; i < n; i++) {
      rain.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        l: 8 + Math.random() * 12,
        s: 4 + Math.random() * 6
      });
    }
  }
  window.addEventListener("resize", resize);
  resize();

  function tick(t) {
    if (!ctx || motion === "off" || reduced) {
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#05070e";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      requestAnimationFrame(tick);
      return;
    }
    ctx.fillStyle = "rgba(5,7,14,0.28)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var g = ctx.createLinearGradient(0, 0, 0, canvas.height);
    g.addColorStop(0, "rgba(18,40,70,0.25)");
    g.addColorStop(1, "rgba(8,12,22,0.5)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(170,200,230,0.18)";
    ctx.lineWidth = 1;
    rain.forEach(function (d) {
      ctx.beginPath();
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(d.x - 2, d.y + d.l);
      ctx.stroke();
      d.y += d.s;
      d.x -= 0.6;
      if (d.y > canvas.height) { d.y = -10; d.x = Math.random() * canvas.width; }
    });
    if (t - lastFlash > 2800 + Math.random() * 4200) {
      lastFlash = t;
      flash = 0.55;
    }
    if (flash > 0.01) {
      ctx.fillStyle = "rgba(210,230,255," + flash + ")";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      flash *= 0.82;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  var mot = $("motion");
  if (mot) {
    if (reduced) { mot.value = "reduced"; motion = "reduced"; }
    mot.addEventListener("change", function () {
      motion = mot.value;
      resize();
    });
  }

  /* ---------------- landing ---------------- */
  var landing = $("landing");
  var app = $("app");
  var initLis = landing ? landing.querySelectorAll(".land-init li") : [];
  var step = 0;
  var entered = false;
  var bar = $("init-bar");
  var initTimer = setInterval(function () {
    if (initLis[step]) initLis[step].classList.add("on");
    if (bar) bar.style.width = Math.min(100, (step + 1) * 25) + "%";
    step += 1;
    if (step >= initLis.length) clearInterval(initTimer);
  }, 650);

  function enterApp() {
    if (entered) return;
    entered = true;
    document.body.classList.remove("intro-lock");
    if (!landing) {
      if (app) app.removeAttribute("hidden");
      return;
    }
    landing.classList.add("gone");
    setTimeout(function () {
      landing.classList.add("hidden");
      if (app) app.removeAttribute("hidden");
      if (indiaMap && indiaMap._resize) indiaMap._resize();
    }, 720);
    try { sessionStorage.setItem("tw_seen_intro", "1"); } catch (e) {}
  }

  if ($("skip-intro")) $("skip-intro").addEventListener("click", enterApp);
  var autoMs = 4200;
  try {
    if (sessionStorage.getItem("tw_seen_intro") === "1") autoMs = 1600;
  } catch (e) {}
  if (reduced) autoMs = 200;
  setTimeout(enterApp, autoMs);

  if ($("replay-intro")) {
    $("replay-intro").addEventListener("click", function () {
      try { sessionStorage.removeItem("tw_seen_intro"); } catch (e) {}
      location.reload();
    });
  }

  /* ---------------- nav ---------------- */
  function setChrome(mode) {
    appMode = mode;
    var modePill = $("mode-pill");
    var sysMode = $("sys-mode");
    var sysSrc = $("sys-source");
    var foot = $("app-foot");
    if (mode === "live") {
      if (modePill) modePill.textContent = "RESEARCH LIVE";
      if (sysMode) sysMode.textContent = "Historical Replay / Live Research";
      if (sysSrc) sysSrc.textContent = "Open-Meteo";
      if (foot) foot.textContent = "FANTASTIC6 · SIH26072 · THUNDERWATCH AI · Research Prototype · LIVE RESEARCH";
      document.body.classList.add("mode-live");
      document.body.classList.remove("mode-replay");
    } else {
      if (modePill) modePill.textContent = "HISTORICAL REPLAY";
      if (sysMode) sysMode.textContent = "Historical Replay / Live Research";
      if (sysSrc) sysSrc.textContent = "Validated historical overlap";
      if (foot) foot.textContent = "FANTASTIC6 · SIH26072 · THUNDERWATCH AI · Research Prototype · HISTORICAL REPLAY";
      document.body.classList.add("mode-replay");
      document.body.classList.remove("mode-live");
    }
  }

  function showView(name) {
    if (name === "overview") name = "replay";
    document.querySelectorAll(".view").forEach(function (v) {
      v.classList.toggle("active", v.getAttribute("data-view") === name);
    });
    document.querySelectorAll(".nav-item").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-view") === name);
    });
    document.querySelectorAll(".mob-nav button").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-view") === name);
    });
    try { history.replaceState(null, "", "#" + name); } catch (e) {}
    if (name === "live") {
      setChrome("live");
      initLiveMap();
      renderLiveTable();
      if (liveMap && liveMap._resize) liveMap._resize();
      if (!liveAllDone && !liveRunning) requestLiveAll();
    } else if (name === "replay") {
      setChrome("replay");
      if (lastPayload) renderAll(lastPayload, { autoFocus: false });
      if (indiaMap && indiaMap._resize) indiaMap._resize();
    }
  }
  document.querySelectorAll(".nav-item, .mob-nav button").forEach(function (b) {
    b.addEventListener("click", function () { showView(b.getAttribute("data-view")); });
  });

  /* ---------------- map ---------------- */
  var indiaMap = null;
  var pendingPayload = null;
  function bindMap(inst) {
    indiaMap = inst;
    if (pendingPayload && indiaMap && indiaMap.setResults) indiaMap.setResults(pendingPayload);
  }
  function startFallbackMap() {
    var mode = $("map-mode");
    if (mode) mode.textContent = "Fallback geographic view";
    if (!window.IndiaMap || !$("map-wrap")) return;
    var inst = new window.IndiaMap();
    inst.init($("map-wrap"), {
      reduced: reduced,
      onSelect: function (sid) { selectStation(sid); }
    });
    bindMap(inst);
  }
  function startGeoMap(cfg) {
    var mode = $("map-mode");
    if (mode) mode.textContent = "MapTiler geographic basemap";
    var inst = new window.TWGeoMap();
    var ok = inst.init($("map-wrap"), {
      reduced: reduced,
      config: cfg,
      onSelect: function (sid) { selectStation(sid); },
      onFail: function () {
        if (inst.map) { try { inst.map.remove(); } catch (e) {} }
        startFallbackMap();
      }
    });
    if (!ok) startFallbackMap();
    else bindMap(inst);
  }
  if ($("map-wrap")) {
    fetch("/api/map-config", { headers: { Accept: "application/json" } })
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        if (cfg && cfg.enabled && window.maplibregl && window.TWGeoMap) startGeoMap(cfg);
        else startFallbackMap();
      })
      .catch(function () { startFallbackMap(); });
  }

  function drawStations(payload, autoFocus) {
    pendingPayload = payload;
    if (indiaMap && indiaMap.setResults) {
      indiaMap.setResults(payload, { autoFocus: !!autoFocus });
    }
  }

  var liveMapCfg = null;
  function bindLiveMap(inst) {
    liveMap = inst;
    if (liveMapPending && liveMap && liveMap.setResults) liveMap.setResults(liveMapPending);
  }
  function startLiveFallback() {
    var modeEl = $("live-map-mode");
    if (modeEl) {
      modeEl.textContent = (liveMapCfg && liveMapCfg.enabled)
        ? "Satellite Hybrid unavailable · fallback geographic view"
        : "MAPTILER_KEY_REQUIRED · fallback geographic view";
    }
    if (!window.IndiaMap || !$("live-map-wrap") || liveMap) return;
    var inst = new window.IndiaMap();
    inst.init($("live-map-wrap"), {
      reduced: reduced,
      onSelect: function (sid) { selectStation(sid); }
    });
    bindLiveMap(inst);
  }
  function startLiveGeo(cfg) {
    if (!$("live-map-wrap") || liveMap) return;
    var modeEl = $("live-map-mode");
    if (modeEl) modeEl.textContent = "MapTiler Satellite Hybrid";
    var inst = new window.TWGeoMap();
    var ok = inst.init($("live-map-wrap"), {
      reduced: reduced,
      config: cfg,
      styleMode: (cfg.styles && cfg.styles.hybrid) ? "hybrid" : "dark",
      onSelect: function (sid) { selectStation(sid); },
      onFail: function () {
        if (inst.map) { try { inst.map.remove(); } catch (e) {} }
        liveMap = null;
        startLiveFallback();
      }
    });
    if (!ok) startLiveFallback();
    else bindLiveMap(inst);
  }
  function initLiveMap() {
    if (liveMap || !$("live-map-wrap")) return;
    if (liveMapCfg && liveMapCfg.enabled && window.maplibregl && window.TWGeoMap) startLiveGeo(liveMapCfg);
    else startLiveFallback();
  }
  fetch("/api/map-config", { headers: { Accept: "application/json" } })
    .then(function (r) { return r.json(); })
    .then(function (cfg) { liveMapCfg = cfg; })
    .catch(function () { liveMapCfg = { enabled: false }; });

  function selectStation(sid) {
    if (!sid) return;
    focusId = sid;
    if (indiaMap && indiaMap.focusStation) indiaMap.focusStation(sid, true);
    if (appMode === "live") {
      var rec = liveByStation[sid];
      if (rec && rec.ok) renderLiveFocus(rec);
      renderLiveTable();
      if (liveMap && liveMap.focusStation) liveMap.focusStation(sid, true);
      return;
    }
    if (!lastPayload) return;
    renderAll(lastPayload, { autoFocus: false });
  }

  function showDashError(msg) {
    var de = $("dash-error");
    var pe = $("page-error");
    if (de) {
      de.textContent = msg;
      de.classList.remove("hidden");
    }
    if (pe) {
      pe.textContent = msg;
      pe.classList.remove("hidden");
    }
  }

  function clearDashError() {
    var de = $("dash-error");
    var pe = $("page-error");
    if (de) { de.textContent = ""; de.classList.add("hidden"); }
    if (pe) { pe.textContent = ""; pe.classList.add("hidden"); }
  }

  /* ---------------- replay ---------------- */
  function setRing(id, pct) {
    var node = $(id);
    if (!node) return;
    var wrap = node.parentElement;
    var arc = wrap.querySelector(".arc");
    var p = pct == null ? 0 : Number(pct);
    node.textContent = pct == null ? "--" : p.toFixed(1) + "%";
    if (arc) {
      var c = 2 * Math.PI * 32;
      arc.style.strokeDasharray = String(c);
      arc.style.strokeDashoffset = String(c * (1 - Math.max(0, Math.min(1, p / 100))));
    }
  }

  function countUp(el, to) {
    if (to == null) { el.textContent = "--"; return; }
    var start = 0, t0 = performance.now();
    function step(now) {
      var k = Math.min(1, (now - t0) / 900);
      el.textContent = (start + (to - start) * k).toFixed(1) + "%";
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll(".demo").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $("hist-time").value = toLocalInput(btn.dataset.ts);
      runAll();
    });
  });
  document.addEventListener("click", function (ev) {
    var t = ev.target && ev.target.closest && ev.target.closest("#run-all");
    if (!t) return;
    ev.preventDefault();
    runAll(ev);
  });
  console.info("[TW] replay button wired");

  var scanMsgs = [
    "HISTORICAL REPLAY PROCESSING — Processing the five reference locations... This may take several minutes. Please wait..."
  ];

  async function runAll(ev) {
    if (ev && ev.preventDefault) ev.preventDefault();
    var timeEl = $("hist-time");
    var raw = isoFromLocal(timeEl ? timeEl.value : "");
    var btn = $("run-all");
    console.info("[TW] runAll timestamp", raw);
    clearDashError();
    if (!raw) {
      showDashError("Please enter a valid historical replay time (UTC).");
      return;
    }
    var scan = $("scan");
    if (scan) {
      scan.classList.remove("hidden");
      scan.textContent = scanMsgs[0];
    }
    var i = 0;
    var iv = setInterval(function () {
      i = Math.min(i + 1, scanMsgs.length - 1);
      if (scan) scan.textContent = scanMsgs[i];
    }, 280);
    if (btn) btn.disabled = true;
    try {
      console.info("[TW] POST /replay/all start");
      var res = await fetch("/replay/all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timestamp_utc: raw })
      });
      var data = await res.json().catch(function () { return {}; });
      console.info("[TW] POST /replay/all status", res.status);
      if (!res.ok) {
        showDashError(data.error || "REPLAY ERROR. Replay could not be completed.");
        if (!lastPayload) {
          $("hero-loc").textContent = "REPLAY ERROR";
          $("hero-pct").textContent = "--";
          $("hero-state").textContent = "TRY AGAIN";
        }
        return;
      }
      lastPayload = data;
      focusId = data.focus_station_id;
      lastSatellite = {};
      try {
        var spat = await fetch("/api/spatial/replay", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ timestamp_utc: raw })
        });
        var spatData = await spat.json().catch(function () { return {}; });
        if (spat.ok && spatData && spatData.satellite_context) {
          lastSatellite = spatData.satellite_context;
        }
      } catch (eSat) {
        lastSatellite = {};
      }
      var tIst = $("replay-t-ist");
      if (tIst) tIst.textContent = "Replay T: " + fmtIst(raw) + " (input stored as UTC)";
      setChrome("replay");
      renderAll(data, { autoFocus: true });
      if (scan) scan.textContent = "HISTORICAL REPLAY READY";
      showView("replay");
    } catch (err) {
      showDashError("REPLAY ERROR. Replay could not be completed.");
      if (!lastPayload) {
        $("hero-loc").textContent = "REPLAY ERROR";
        $("hero-pct").textContent = "--";
        $("hero-state").textContent = "TRY AGAIN";
      }
    } finally {
      clearInterval(iv);
      if (btn) btn.disabled = false;
    }
  }

  function renderAll(data, opts) {
    opts = opts || {};
    drawStations(data, opts.autoFocus === true);
    var body = $("loc-body");
    if (!body) return;
    body.innerHTML = "";
    (data.stations || []).forEach(function (s, idx) {
      var tr = document.createElement("tr");
      if (idx === 0 && !s.unavailable) tr.className = "top";
      if (s.station_id === focusId) tr.style.outline = "1px solid #4ec4ff";
      tr.innerHTML = s.unavailable
        ? "<td>" + (idx + 1) + "</td><td>" + s.station_name + " <span class='icao'>" + s.station_id + "</span></td><td>—</td><td>—</td><td>—</td><td>UNAVAILABLE</td>"
        : "<td>" + (idx + 1) + "</td><td>" + s.station_name + " <span class='icao'>" + s.station_id + "</span></td><td>" + s.lead_1h_pct + "%</td><td>" + s.lead_2h_pct + "%</td><td>" + s.lead_3h_pct + "%</td><td>" + (s.threshold_state || "") + "</td>";
      tr.addEventListener("click", function () { selectStation(s.station_id); });
      body.appendChild(tr);
    });

    if (data.all_unavailable) {
      $("hero-loc").textContent = "REPLAY UNAVAILABLE";
      $("hero-icao").textContent = "";
      $("hero-pct").textContent = "--";
      $("hero-state").textContent = "Required historical inputs were unavailable for this timestamp.";
      setRing("r1", null); setRing("r2", null); setRing("r3", null);
      $("atmo-grid").innerHTML = "<p class='muted'>UNAVAILABLE</p>";
      if ($("verify-body")) $("verify-body").innerHTML = "<p class='muted'>No verification — replay unavailable.</p>";
      renderSatellite(null);
      return;
    }
    var s = (data.stations || []).find(function (x) { return x.station_id === focusId; }) || data.stations[0];
    renderFocus(s, data);
  }

  function renderFocus(s, data) {
    if (!s) return;
    if (s.unavailable) {
      $("hero-loc").textContent = s.station_name;
      $("hero-icao").textContent = s.station_id;
      $("hero-pct").textContent = "--";
      $("hero-state").textContent = "UNAVAILABLE";
      setRing("r1", null); setRing("r2", null); setRing("r3", null);
      $("atmo-grid").innerHTML = "<p class='muted'>UNAVAILABLE</p>";
      if ($("verify-body")) $("verify-body").innerHTML = "<p class='muted'>UNAVAILABLE</p>";
      return;
    }
    var isHighest = data && s.station_id === data.focus_station_id;
    $("hero-loc").textContent = s.station_name;
    $("hero-icao").textContent = s.station_id;
    countUp($("hero-pct"), Number(s.lead_1h_pct));
    $("hero-thr").textContent = "Threshold: " + (s.threshold_pct || "6.5%");
    $("hero-state").textContent = s.threshold_state || "";
    $("hero-state").className = "state " + (s.threshold_state === "ABOVE THRESHOLD" ? "above" : "below");
    var kick = document.querySelector("#hero-risk .kicker");
    if (kick) kick.textContent = isHighest ? "HIGHEST MODEL RISK" : "FOCUSED LOCATION";
    setRing("r1", s.lead_1h_pct == null ? null : Number(s.lead_1h_pct));
    setRing("r2", s.lead_2h_pct == null ? null : Number(s.lead_2h_pct));
    setRing("r3", s.lead_3h_pct == null ? null : Number(s.lead_3h_pct));

    var ag = $("atmo-grid");
    if (!s.atmosphere || !s.atmosphere.length) {
      ag.innerHTML = "<p class='muted'>DATA UNAVAILABLE</p>";
    } else {
      ag.innerHTML = s.atmosphere.map(function (a) {
        return "<div><span>" + a.label + "</span><strong>" + a.value + " " + a.unit + "</strong></div>";
      }).join("");
    }
    $("verify-body").innerHTML =
      "<ul>" +
      "<li>+1h model " + s.lead_1h_pct + "% (" + s.alert_1h + ") → " + s.target_1h_text + (s.hit_1h ? " · " + s.hit_1h : "") + "</li>" +
      "<li>+2h model " + s.lead_2h_pct + "% → " + s.target_2h_text + (s.hit_2h ? " · " + s.hit_2h : "") + "</li>" +
      "<li>+3h model " + s.lead_3h_pct + "% → " + s.target_3h_text + (s.hit_3h ? " · " + s.hit_3h : "") + "</li>" +
      "</ul>";
    renderSatellite(s.station_id);
  }

  function renderSatellite(sid) {
    var box = $("sat-grid");
    if (!box) return;
    if (appMode === "live") {
      box.innerHTML = "<p class='muted'>Satellite case replay is historical only. Not used in live Model B.</p>";
      return;
    }
    var rec = lastSatellite && sid ? lastSatellite[sid] : null;
    if (!rec) {
      box.innerHTML = "<p class='muted'>No INSAT-3DR CMK observation for this hour. Missing satellite is not treated as clear.</p>";
      return;
    }
    function cell(label, val, unit) {
      var show = val == null || val === "" || !isFinite(Number(val));
      return "<div><span>" + label + "</span><strong>" + (show ? "NOT AVAILABLE" : Number(val).toFixed(unit === "" ? 0 : 2) + (unit ? " " + unit : "")) + "</strong></div>";
    }
    var mask = rec.sat_cloud_mask_at_station;
    var maskTxt = mask == null ? "NOT AVAILABLE" : (Number(mask) >= 0.5 ? "CLOUDY" : "CLEAR");
    box.innerHTML =
      "<div><span>Cloud mask</span><strong>" + maskTxt + "</strong></div>" +
      cell("Cloudy fraction", rec.sat_cloudy_fraction_25km, "") +
      cell("Clear fraction", rec.sat_clear_fraction_25km, "") +
      cell("Valid pixels", rec.sat_valid_pixel_count_25km, "") +
      "<div><span>Observation</span><strong>" + (rec.sat_observation_time_utc ? fmtIst(rec.sat_observation_time_utc) : "NOT AVAILABLE") + "</strong></div>" +
      cell("Age", rec.sat_age_minutes, "min") +
      "<p class='muted'>Evidence only — does not change Model B probability.</p>";
  }

  function pctFromProb(p) {
    if (p == null || p === "") return null;
    var n = Number(p);
    if (!isFinite(n)) return null;
    return n * 100;
  }

  function fmtIst(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "—";
    var text = new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata",
      hour: "numeric",
      minute: "2-digit",
      hour12: true
    }).format(d);
    return text + " IST";
  }

  function liveErrorText(code) {
    return "LIVE PREDICTION UNAVAILABLE. Required atmospheric data could not be obtained safely. No prediction was generated.";
  }

  function clearLiveCards() {
    ["live-p1", "live-p2", "live-p3"].forEach(function (id) {
      if ($(id)) $(id).textContent = "--";
    });
    ["live-a1", "live-a2", "live-a3"].forEach(function (id) {
      if ($(id)) $(id).textContent = "—";
    });
    document.querySelectorAll(".live-lead").forEach(function (el) {
      el.classList.remove("alert", "ok");
      el.classList.add("cleared");
    });
    if ($("live-src")) $("live-src").textContent = "—";
    if ($("live-obs")) $("live-obs").textContent = "—";
    if ($("live-ptime")) $("live-ptime").textContent = "—";
    if ($("live-age")) $("live-age").textContent = "—";
    if ($("live-v1")) $("live-v1").textContent = "valid —";
    if ($("live-v2")) $("live-v2").textContent = "valid —";
    if ($("live-v3")) $("live-v3").textContent = "valid —";
  }

  function paintLiveLead(pid, aid, block, leadEl) {
    if (!block || block.probability == null) {
      if ($(pid)) $(pid).textContent = "--";
      if ($(aid)) $(aid).textContent = "—";
      if (leadEl) { leadEl.classList.remove("alert", "ok"); leadEl.classList.add("cleared"); }
      return;
    }
    var pct = pctFromProb(block.probability);
    if ($(pid)) $(pid).textContent = pct == null ? "--" : pct.toFixed(1) + "%";
    var alert = !!block.alert;
    if ($(aid)) $(aid).textContent = alert ? "ALERT" : "NO ALERT";
    if (leadEl) {
      leadEl.classList.remove("cleared");
      leadEl.classList.toggle("alert", alert);
      leadEl.classList.toggle("ok", !alert);
    }
  }

  function renderLiveFocus(data) {
    var preds = data.predictions || {};
    paintLiveLead("live-p1", "live-a1", preds["1h"], document.querySelector('.live-lead[data-lead="1h"]'));
    paintLiveLead("live-p2", "live-a2", preds["2h"], document.querySelector('.live-lead[data-lead="2h"]'));
    paintLiveLead("live-p3", "live-a3", preds["3h"], document.querySelector('.live-lead[data-lead="3h"]'));
    var src = data.data_source || "Open-Meteo";
    if ($("live-src")) $("live-src").textContent = /open-meteo/i.test(src) ? "Open-Meteo" : src;
    if ($("live-obs")) $("live-obs").textContent = fmtIst(data.data_observation_time_utc);
    if ($("live-ptime")) $("live-ptime").textContent = fmtIst(data.prediction_time_utc);
    if ($("live-age")) {
      var age = data.data_age_minutes;
      $("live-age").textContent = age == null ? "—" : Math.round(Number(age)) + " min";
    }
    if ($("live-mode")) $("live-mode").textContent = "LIVE";
    var vt = data.forecast_valid_times_utc || {};
    if ($("live-v1")) $("live-v1").textContent = vt["1h"] ? "valid " + fmtIst(vt["1h"]) : "valid —";
    if ($("live-v2")) $("live-v2").textContent = vt["2h"] ? "valid " + fmtIst(vt["2h"]) : "valid —";
    if ($("live-v3")) $("live-v3").textContent = vt["3h"] ? "valid " + fmtIst(vt["3h"]) : "valid —";
    var sat = $("live-sat-grid");
    if (sat) sat.innerHTML = "<p class='muted'>Satellite case replay is historical only. Not used in live Model B.</p>";

    var ag = $("live-atmo-grid");
    var sig = data.signals || {};
    if (ag) {
      ag.innerHTML = SIGNAL_LABELS.map(function (row) {
        var val = sig[row[0]];
        if (val == null || val === "" || !isFinite(Number(val))) {
          return "<div><span>" + row[1] + "</span><strong>NOT AVAILABLE</strong></div>";
        }
        return "<div><span>" + row[1] + "</span><strong>" + Number(val).toFixed(1) + " " + row[2] + "</strong></div>";
      }).join("");
    }
  }

  function liveMapPayload() {
    var names = window.TW && window.TW.stations ? window.TW.stations : {};
    var stations = LIVE_ORDER.map(function (sid) {
      var rec = liveByStation[sid];
      if (!rec || !rec.ok) {
        return {
          station_id: sid,
          station_name: names[sid] || sid,
          unavailable: true,
          live_status: rec && rec.status ? rec.status : "NOT QUERIED",
          lead_1h_probability: null,
          lead_1h_pct: null
        };
      }
      var p1 = rec.predictions && rec.predictions["1h"] ? rec.predictions["1h"].probability : null;
      var p2 = rec.predictions && rec.predictions["2h"] ? rec.predictions["2h"].probability : null;
      var p3 = rec.predictions && rec.predictions["3h"] ? rec.predictions["3h"].probability : null;
      return {
        station_id: sid,
        station_name: names[sid] || sid,
        unavailable: false,
        live_status: "LIVE",
        lead_1h_probability: p1,
        lead_2h_probability: p2,
        lead_3h_probability: p3,
        lead_1h_pct: pctFromProb(p1) == null ? null : pctFromProb(p1).toFixed(1),
        lead_2h_pct: pctFromProb(p2) == null ? null : pctFromProb(p2).toFixed(1),
        lead_3h_pct: pctFromProb(p3) == null ? null : pctFromProb(p3).toFixed(1)
      };
    });
    return { stations: stations, focus_station_id: focusId, replay_mode: "LIVE" };
  }

  function liveSortedIds() {
    var lead = liveSortLead || "1h";
    return LIVE_ORDER.slice().sort(function (a, b) {
      var ra = liveByStation[a], rb = liveByStation[b];
      var pa = ra && ra.ok && ra.predictions && ra.predictions[lead] ? Number(ra.predictions[lead].probability) : -1;
      var pb = rb && rb.ok && rb.predictions && rb.predictions[lead] ? Number(rb.predictions[lead].probability) : -1;
      if (pa !== pb) return pb - pa;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  function updateLiveHero() {
    var names = window.TW && window.TW.stations ? window.TW.stations : {};
    var ids = liveSortedIds();
    var top = null;
    for (var i = 0; i < ids.length; i++) {
      if (liveByStation[ids[i]] && liveByStation[ids[i]].ok) { top = liveByStation[ids[i]]; break; }
    }
    if (!top) {
      if ($("live-hero-loc")) $("live-hero-loc").textContent = "AWAITING LIVE DATA";
      if ($("live-hero-icao")) $("live-hero-icao").textContent = "—";
      if ($("live-hero-pct")) $("live-hero-pct").textContent = "--";
      return;
    }
    if ($("live-hero-loc")) $("live-hero-loc").textContent = names[top.station_id] || top.station_id;
    if ($("live-hero-icao")) $("live-hero-icao").textContent = top.station_id;
    var p1 = pctFromProb(top.predictions["1h"] && top.predictions["1h"].probability);
    if ($("live-hero-pct")) $("live-hero-pct").textContent = p1 == null ? "--" : p1.toFixed(1) + "%";
    var a1 = top.predictions["1h"] && top.predictions["1h"].alert;
    if ($("live-hero-state")) {
      $("live-hero-state").textContent = a1 ? "ALERT" : "NO ALERT";
      $("live-hero-state").className = "state " + (a1 ? "above" : "below");
    }
  }

  function renderLiveStorm() {
    var box = $("live-storm");
    if (!box) return;
    if (liveRunning && !liveAllDone) {
      box.innerHTML = "<p class='proc-note'>LIVE DATA PROCESSING — Fetching current atmospheric and NWP data for 5 locations. This may take several minutes depending on external data availability. Please wait...</p>";
      return;
    }
    var names = window.TW && window.TW.stations ? window.TW.stations : {};
    box.innerHTML = LIVE_ORDER.map(function (sid) {
      var rec = liveByStation[sid];
      var p1 = "--", p2 = "--", p3 = "--", st = liveAllDone ? "UNAVAILABLE" : "AWAITING LIVE DATA";
      var cls = "storm-card";
      if (rec && rec.ok && rec.predictions) {
        var q1 = pctFromProb(rec.predictions["1h"].probability);
        var q2 = pctFromProb(rec.predictions["2h"].probability);
        var q3 = pctFromProb(rec.predictions["3h"].probability);
        p1 = q1 == null ? "--" : q1.toFixed(1) + "%";
        p2 = q2 == null ? "--" : q2.toFixed(1) + "%";
        p3 = q3 == null ? "--" : q3.toFixed(1) + "%";
        st = rec.predictions["1h"].alert ? "ALERT" : "NO ALERT";
        cls += rec.predictions["1h"].alert ? " alert" : " ok";
      }
      return "<button type='button' class='" + cls + "' data-sid='" + sid + "'><strong>" + (names[sid] || sid) + "</strong><span class='icao'>" + sid + "</span><span>+1H " + p1 + "</span><span>+2H " + p2 + "</span><span>+3H " + p3 + "</span><em>6.5% · " + st + "</em></button>";
    }).join("");
    box.querySelectorAll(".storm-card").forEach(function (btn) {
      btn.addEventListener("click", function () { selectStation(btn.getAttribute("data-sid")); });
    });
  }

  function renderLiveTable() {
    var names = window.TW && window.TW.stations ? window.TW.stations : {};
    var order = liveSortedIds();
    function fill(body) {
      if (!body) return;
      body.innerHTML = "";
      order.forEach(function (sid, idx) {
        var rec = liveByStation[sid];
        var tr = document.createElement("tr");
        if (sid === focusId) tr.style.outline = "1px solid #4ec4ff";
        var status = liveAllDone ? "UNAVAILABLE" : "AWAITING LIVE DATA";
        var c1 = "—", c2 = "—", c3 = "—";
        if (rec && rec.ok && rec.predictions) {
          status = "LIVE";
          var q1 = pctFromProb(rec.predictions["1h"].probability);
          var q2 = pctFromProb(rec.predictions["2h"].probability);
          var q3 = pctFromProb(rec.predictions["3h"].probability);
          c1 = q1 == null ? "—" : q1.toFixed(1) + "%";
          c2 = q2 == null ? "—" : q2.toFixed(1) + "%";
          c3 = q3 == null ? "—" : q3.toFixed(1) + "%";
        } else if (rec && rec.ok === false) {
          status = "UNAVAILABLE";
        }
        tr.innerHTML = "<td>" + (idx + 1) + "</td><td>" + (names[sid] || sid) + " <span class='icao'>" + sid + "</span></td><td>" + c1 + "</td><td>" + c2 + "</td><td>" + c3 + "</td><td>" + status + "</td>";
        tr.addEventListener("click", function () { selectStation(sid); });
        body.appendChild(tr);
      });
    }
    fill($("live-loc-body"));
    renderLiveStorm();
    var payload = liveMapPayload();
    liveMapPending = payload;
    if (liveMap && liveMap.setResults) liveMap.setResults(payload, { autoFocus: false });
    updateLiveHero();
    if (focusId && liveByStation[focusId] && liveByStation[focusId].ok) renderLiveFocus(liveByStation[focusId]);
  }

  async function requestLiveAll() {
    if (liveRunning) return;
    liveRunning = true;
    var token = ++liveFetchToken;
    var btn = $("run-live-all");
    var fetchEl = $("live-fetch");
    var errEl = $("live-error");
    if (btn) { btn.disabled = true; btn.textContent = "PROCESSING..."; }
    if (errEl) { errEl.textContent = ""; errEl.classList.add("hidden"); }
    clearLiveCards();
    if (fetchEl) {
      fetchEl.textContent = "LIVE DATA PROCESSING — Fetching current atmospheric and NWP data for 5 locations. This may take several minutes depending on external data availability. Please wait... PROCESSING 5 LOCATIONS";
    }
    renderLiveStorm();
    try {
      var res = await fetch("/api/inference/live/all", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ stations: LIVE_ORDER })
      });
      var data = await res.json().catch(function () { return {}; });
      if (token !== liveFetchToken) return;
      liveAllDone = true;
      (data.results || []).forEach(function (row) {
        if (row.status === "LIVE" && row.predictions) {
          liveByStation[row.station_id] = {
            ok: true,
            status: "LIVE",
            station_id: row.station_id,
            predictions: row.predictions,
            data_observation_time_utc: row.observation_time_utc,
            prediction_time_utc: row.prediction_time_utc,
            forecast_valid_times_utc: row.forecast_valid_times_utc,
            data_age_minutes: row.data_age_minutes,
            data_source: row.source,
            signals: row.signals,
            feature_validation: row.feature_validation
          };
        } else {
          liveByStation[row.station_id] = {
            ok: false,
            status: row.error_code || "LIVE_DATA_UNAVAILABLE",
            station_id: row.station_id
          };
        }
      });
      var avail = data.available_stations != null ? data.available_stations : 0;
      var reqn = data.requested_stations || 5;
      if (avail === 0) {
        if (fetchEl) fetchEl.textContent = "LIVE DATA UNAVAILABLE · 0 / " + reqn + " locations available. No prediction was generated.";
        if (errEl) {
          errEl.textContent = "External atmospheric/NWP data could not be obtained safely.";
          errEl.classList.remove("hidden");
        }
      } else if (avail < reqn) {
        if (fetchEl) fetchEl.textContent = "LIVE DATA READY · " + avail + " / " + reqn + " LOCATIONS AVAILABLE";
      } else {
        if (fetchEl) fetchEl.textContent = "LIVE DATA READY · " + avail + " / " + reqn + " LOCATIONS AVAILABLE";
      }
      var ids = liveSortedIds();
      focusId = ids[0];
      renderLiveTable();
    } catch (e) {
      if (token !== liveFetchToken) return;
      liveAllDone = true;
      if (fetchEl) fetchEl.textContent = "LIVE DATA UNAVAILABLE";
      if (errEl) {
        errEl.textContent = liveErrorText("LIVE_DATA_UNAVAILABLE");
        errEl.classList.remove("hidden");
      }
      renderLiveTable();
    } finally {
      liveRunning = false;
      if (btn) { btn.disabled = false; btn.textContent = "RUN 5-LOCATION LIVE DATA"; }
    }
  }

  if ($("run-live-all")) {
    $("run-live-all").addEventListener("click", function () {
      liveAllDone = false;
      requestLiveAll();
    });
  }
  document.querySelectorAll(".live-sort-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      liveSortLead = b.getAttribute("data-lead") || "1h";
      document.querySelectorAll(".live-sort-btn").forEach(function (x) {
        x.classList.toggle("active", x === b);
      });
      if ($("live-sort-lab")) $("live-sort-lab").textContent = "SORTED BY +" + liveSortLead.toUpperCase() + " MODEL RISK";
      renderLiveTable();
    });
  });

  var initialView = "live";
  try {
    var h = String(location.hash || "").replace("#", "");
    if (h === "overview") h = "replay";
    if (h) initialView = h;
  } catch (eHash) {}
  showView(initialView);
})();
