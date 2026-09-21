(function () {
  "use strict";

  var lastPayload = null;
  var focusId = null;
  var motion = "full";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
  function showView(name) {
    document.querySelectorAll(".view").forEach(function (v) {
      v.classList.toggle("active", v.getAttribute("data-view") === name);
    });
    document.querySelectorAll(".nav-item").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-view") === name);
    });
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

  function selectStation(sid) {
    if (!sid) return;
    focusId = sid;
    if (indiaMap && indiaMap.focusStation) indiaMap.focusStation(sid, true);
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
    "SCANNING LOCATIONS",
    "BUILDING FEATURE STATE",
    "RUNNING MODEL",
    "COMPARING LOCATIONS",
    "IDENTIFYING HIGHEST RISK"
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
      renderAll(data, { autoFocus: true });
      showView("overview");
    } catch (err) {
      showDashError("REPLAY ERROR. Replay could not be completed.");
      if (!lastPayload) {
        $("hero-loc").textContent = "REPLAY ERROR";
        $("hero-pct").textContent = "--";
        $("hero-state").textContent = "TRY AGAIN";
      }
    } finally {
      clearInterval(iv);
      if (scan) scan.classList.add("hidden");
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
    if (!isHighest) {
      document.querySelector(".hero-risk .kicker").textContent = "FOCUSED LOCATION";
    } else {
      document.querySelector(".hero-risk .kicker").textContent = "HIGHEST MODEL RISK";
    }
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
  }
})();
