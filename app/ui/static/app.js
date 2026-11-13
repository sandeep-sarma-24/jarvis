"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const chat = $("chat");
  const MAX_LINES = 500, MAX_TOOLS = 100, AVG_WINDOW = 20;
  let ws, retry = 0, attached = null, useCam = false, since = Date.now();
  let turns = 0, hello = false;
  const latencies = [];
  const bubbles = new Map(); // turn_id -> assistant bubble element

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  const stamp = (ts) => {
    const d = new Date((ts || Date.now() / 1000) * 1000);
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
  };

  // ---- scrolling panes: autoscroll unless the user scrolled up ----
  const stuck = new WeakMap();
  function follow(box) {
    stuck.set(box, true);
    box.addEventListener("scroll", () => stuck.set(box, box.scrollHeight - box.scrollTop - box.clientHeight < 30));
  }
  const toEnd = (box) => { if (stuck.get(box)) box.scrollTop = box.scrollHeight; };
  [chat, $("toollist"), ...document.querySelectorAll(".log")].forEach(follow);

  function append(box, node, cap) {
    box.appendChild(node);
    while (box.childElementCount > cap) box.firstElementChild.remove();
    toEnd(box);
  }

  // ---- pipeline logs ----
  const paneFor = { stt: "stt", llm: "llm", tts: "tts" };
  function logLine(channel, level, text, ts) {
    const line = el("div", "line " + level);
    line.appendChild(el("span", "t", `[${stamp(ts)}] `));
    line.appendChild(el("span", "ch", channel.toUpperCase().padEnd(6)));
    line.appendChild(el("span", "lv", level.padEnd(5) + " "));
    line.appendChild(el("span", "x", text));
    append($("log-" + (paneFor[channel] || "sys")), line, MAX_LINES);
  }

  // ---- tools ----
  const SCREEN_TOOLS = new Set(["look_at_screen", "find_element", "press_element", "type_into"]);
  function onTool(m) {
    if ($("toollist").querySelector(".empty")) $("toollist").textContent = "";
    const li = el("li");
    li.appendChild(el("span", m.ok ? "ok" : "fail", `[${stamp(m.ts)}] ${m.ok ? "OK  " : "FAIL"} ${m.name}`));
    li.appendChild(el("span", "args", "args: " + JSON.stringify(m.args == null ? {} : m.args)));
    if (m.result != null) li.appendChild(el("span", "res", "=> " + m.result));
    append($("toollist"), li, MAX_TOOLS);
    if (SCREEN_TOOLS.has(m.name)) {
      $("screen").classList.add("flash");
      setTimeout(() => $("screen").classList.remove("flash"), 1200);
    }
    logLine("tool", m.ok ? "INFO" : "ERROR", `${m.name}(${JSON.stringify(m.args)}) ${m.ok ? "ok" : "failed"}`, m.ts);
  }

  // ---- chat rendering ----
  function bubble(role, text, ts, extra) {
    const b = el("div", "msg " + role);
    if (extra && extra.image) {
      const im = el("img");
      im.alt = "Attached photo";
      im.src = extra.image; // data: URL from server, set as attribute value (not HTML)
      b.appendChild(im);
    }
    b.appendChild(el("span", "txt", text));
    b.appendChild(el("span", "meta", stamp(ts)));
    append(chat, b, MAX_LINES);
    return b;
  }
  const txt = (b) => b.querySelector(".txt");

  function renderHistory(h) {
    chat.textContent = "";
    bubbles.clear();
    for (const m of h || []) bubble(m.role === "user" ? "user" : "assistant", m.text, m.ts);
    stuck.set(chat, true); toEnd(chat);
  }
  function onDelta(m) {
    let b = bubbles.get(m.turn_id);
    if (!b) {
      b = bubble("assistant", "", null);
      b.classList.add("streaming");
      bubbles.set(m.turn_id, b);
    }
    txt(b).textContent += m.text;
    toEnd(chat);
  }
  function onAssistant(m) {
    const b = bubbles.get(m.turn_id) || bubble("assistant", "", m.ts);
    bubbles.delete(m.turn_id);
    b.classList.remove("streaming");
    if (m.outcome === "cancelled" || m.outcome === "failed") b.classList.add(m.outcome);
    if (m.text) txt(b).textContent = m.text; // keep streamed text if final is empty (cancelled)
    const parts = [stamp(m.ts)];
    if (m.latency_ms != null) parts.push(Math.round(m.latency_ms) + " ms");
    if (m.outcome && m.outcome !== "completed") parts.push(m.outcome);
    b.querySelector(".meta").textContent = parts.join(" | ");
    $("turns").textContent = ++turns;
    if (m.latency_ms != null) {
      latencies.push(m.latency_ms);
      if (latencies.length > AVG_WINDOW) latencies.shift();
      $("last").textContent = Math.round(m.latency_ms);
      $("avg").textContent = Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
    }
    toEnd(chat);
  }

  // ---- metrics ----
  function setState(s) {
    $("orb").dataset.state = s;
    $("statelabel").textContent = s.toUpperCase();
  }
  function flag(id, on) {
    const e = $(id);
    e.textContent = on ? "ON" : "OFF";
    e.className = on ? "" : "off";
  }
  function conn(on) {
    const c = $("conn");
    c.textContent = on ? "ONLINE" : "OFFLINE";
    c.className = on ? "" : "off";
  }
  setInterval(() => {
    const s = Math.floor((Date.now() - since) / 1000);
    $("uptime").textContent = `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`;
  }, 1000);

  function onVision(m) {
    const c = $("caption");
    c.textContent = m.text; c.hidden = false;
    logLine("vision", "INFO", m.text, m.ts);
  }
  // ---- detector overlay: boxes on a canvas over the (object-fit: contain) MJPEG image ----
  const cvs = $("boxes"), cg = cvs.getContext("2d");
  let ents = null, entsAt = 0;
  function drawBoxes() {
    const vp = cvs.parentElement, v = $("video"), dpr = devicePixelRatio || 1;
    const W = vp.clientWidth, H = vp.clientHeight;
    if (cvs.width !== W * dpr || cvs.height !== H * dpr) { cvs.width = W * dpr; cvs.height = H * dpr; }
    cg.setTransform(dpr, 0, 0, dpr, 0, 0);
    cg.clearRect(0, 0, W, H);
    if (!ents || !v.naturalWidth) return;
    const k = Math.min(W / v.naturalWidth, H / v.naturalHeight);
    const w = v.naturalWidth * k, h = v.naturalHeight * k, x0 = (W - w) / 2, y0 = (H - h) / 2;
    cg.font = "11px ui-monospace, Menlo, monospace"; cg.textBaseline = "top";
    cg.strokeStyle = cg.fillStyle = "#00ff41"; cg.lineWidth = 1.5; cg.shadowColor = "#00ff41"; cg.shadowBlur = 4;
    for (const e of ents.items) {
      const [a, b, c, d] = e.box, x = x0 + a * w, y = y0 + b * h;
      cg.strokeRect(x, y, (c - a) * w, (d - b) * h);
      const t = `${e.label} #${e.id} ${e.conf.toFixed(2)}`, tw = cg.measureText(t).width + 6;
      const ty = y >= 14 ? y - 14 : y;
      cg.shadowBlur = 0; cg.fillStyle = "rgba(0,0,0,.75)"; cg.fillRect(x, ty, tw, 13);
      cg.fillStyle = "#00ff41"; cg.fillText(t, x + 3, ty + 1); cg.shadowBlur = 4;
    }
  }
  function onEntities(m) {
    ents = m; entsAt = Date.now();
    $("detstat").textContent = `${m.fps.toFixed(1)} / ${Math.round(m.ms)}`;
    $("detstat").className = "";
    $("hudents").textContent = m.summary ? "DET " + m.summary.replace(/ \(.*$/, "") : "";
    drawBoxes();
  }
  setInterval(() => {  // detector went quiet: clear stale boxes
    if (ents && Date.now() - entsAt > 2000) {
      ents = null; drawBoxes();
      $("detstat").textContent = "--"; $("detstat").className = "off"; $("hudents").textContent = "";
    }
  }, 1000);
  addEventListener("resize", drawBoxes);

  function toast(msg) {
    const t = el("div", "toast");
    t.appendChild(el("span", null, msg));
    const x = el("button", null, "x");
    x.type = "button"; x.setAttribute("aria-label", "Dismiss");
    x.onclick = () => t.remove();
    t.appendChild(x);
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 10000);
  }

  function handle(m) {
    switch (m.type) {
      case "hello":
        if (!hello) { hello = true; since = Date.now(); }
        $("model").textContent = m.model || "--";
        $("model").title = m.model || "";
        flag("camflag", m.camera); flag("visflag", m.vision);
        $("hudflags").textContent = `CAM ${m.camera ? "ON" : "OFF"} / VIS ${m.vision ? "ON" : "OFF"}`;
        renderHistory(m.history);
        break;
      case "status": setState(m.state); break;
      case "user": bubble("user", m.text, m.ts, { image: m.image }); break;
      case "assistant_delta": onDelta(m); break;
      case "assistant": onAssistant(m); break;
      case "vision": onVision(m); break;
      case "entities": onEntities(m); break;
      case "log": logLine(m.channel, m.level, m.msg, m.ts); break;
      case "tool": onTool(m); break;
      case "screen": onScreen(m); break;
      case "screen_result": onScreenResult(m); break;
      case "error": toast(m.message || "Error"); break;
    }
  }

  // ---- websocket with backoff ----
  function connect() {
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => { retry = 0; conn(true); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); } };
    ws.onclose = () => {
      conn(false); setState("idle");
      setTimeout(connect, Math.min(10000, 500 * 2 ** retry++));
    };
    ws.onerror = () => ws.close();
  }
  const send = (o) => { if (ws && ws.readyState === 1) { ws.send(JSON.stringify(o)); return true; } toast("Not connected"); return false; };

  // ---- screen (accessibility tree) ----
  const TEXTY = new Set(["textfield", "textarea", "searchfield", "combobox"]);
  let snap = null, byId = new Map(), kids = new Map(), sel = null, collapsed = new Set(), box = null;

  function onScreen(m) {
    const prev = sel && byId.get(sel);
    snap = m;
    byId = new Map(m.elements.map((e) => [e.id, e]));
    kids = new Map();
    for (const e of m.elements) {
      if (!kids.has(e.parent)) kids.set(e.parent, []);
      kids.get(e.parent).push(e);
    }
    // ids are only valid per snapshot: keep the selection only if role+label still match
    const same = prev && byId.get(prev.id);
    sel = same && same.role === prev.role && same.label === prev.label ? prev.id : null;
    renderScreen();
  }
  function renderScreen() {
    const n = snap.elements.length;
    $("scrinfo").textContent = `${snap.app} | ${n} elements${snap.truncated ? " (truncated)" : ""} | ${Math.max(0, Math.round(Date.now() / 1000 - snap.ts))}s ago`;
    renderTree(); renderWire(); renderInspector();
  }
  setInterval(() => { if (snap) renderScreenInfo(); }, 5000);
  function renderScreenInfo() {
    $("scrinfo").textContent = $("scrinfo").textContent.replace(/\| \d+s ago$/, `| ${Math.max(0, Math.round(Date.now() / 1000 - snap.ts))}s ago`);
  }
  function select(id) { sel = id; renderTree(); renderWire(); renderInspector(); }

  function renderTree() {
    const tree = $("scrtree"), top = tree.scrollTop;
    tree.textContent = "";
    const walk = (parent, depth) => {
      for (const e of kids.get(parent) || []) {
        const has = kids.has(e.id), open = !collapsed.has(e.id);
        const row = el("button", "trow" + (e.interactive ? " int" : "") + (e.enabled ? "" : " dis") + (e.id === sel ? " sel" : ""));
        row.type = "button";
        row.style.paddingLeft = 6 + depth * 12 + "px";
        row.setAttribute("role", "treeitem");
        const tg = el("span", "tg", has ? (open ? "\u25be" : "\u25b8") : "");
        if (has) tg.onclick = (ev) => { ev.stopPropagation(); open ? collapsed.add(e.id) : collapsed.delete(e.id); renderTree(); };
        row.appendChild(tg);
        row.appendChild(el("span", "tx", `${e.id} ${e.role}${e.secure ? " \ud83d\udd12" : ""} "${e.label || ""}"`));
        row.onclick = () => select(e.id);
        tree.appendChild(row);
        if (has && open) walk(e.id, depth + 1);
      }
    };
    walk(null, 0);
    tree.scrollTop = top;
  }

  // wireframe: frames normalized to the bounding box of the root elements (windows)
  function renderWire() {
    const w = $("scrwire");
    w.textContent = "";
    const roots = (kids.get(null) || []).filter((e) => e.frame);
    const fr = roots.length ? roots.map((e) => e.frame) : snap.elements.filter((e) => e.frame).map((e) => e.frame);
    if (!fr.length) return;
    const x0 = Math.min(...fr.map((f) => f[0])), y0 = Math.min(...fr.map((f) => f[1]));
    const bw = Math.max(1, Math.max(...fr.map((f) => f[0] + f[2])) - x0);
    const bh = Math.max(1, Math.max(...fr.map((f) => f[1] + f[3])) - y0);
    const k = Math.min(w.clientWidth / bw, w.clientHeight / bh) || 1;
    box = { x0, y0, k };
    for (const e of snap.elements) {
      if (!e.frame) continue;
      const d = el("div", "wf" + (e.interactive ? " int" : "") + (e.id === sel ? " sel" : ""));
      d.dataset.id = e.id;
      Object.assign(d.style, { left: (e.frame[0] - x0) * k + "px", top: (e.frame[1] - y0) * k + "px",
        width: Math.max(1, e.frame[2] * k) + "px", height: Math.max(1, e.frame[3] * k) + "px" });
      w.appendChild(d);
    }
  }
  // click: smallest box under the pointer
  $("scrwire").addEventListener("click", (ev) => {
    if (!snap || !box) return;
    const r = $("scrwire").getBoundingClientRect();
    const x = (ev.clientX - r.left + $("scrwire").scrollLeft) / box.k + box.x0;
    const y = (ev.clientY - r.top + $("scrwire").scrollTop) / box.k + box.y0;
    let best = null, area = Infinity;
    for (const e of snap.elements) {
      const f = e.frame;
      if (f && x >= f[0] && x <= f[0] + f[2] && y >= f[1] && y <= f[1] + f[3] && f[2] * f[3] <= area) { best = e; area = f[2] * f[3]; }
    }
    if (best) select(best.id);
  });
  addEventListener("resize", () => { if (snap) renderWire(); });

  function renderInspector() {
    const e = sel && byId.get(sel), a = $("scrattrs");
    a.textContent = "";
    if (!e) a.textContent = "select an element";
    else {
      const rows = [["id", e.id], ["role", e.role], ["label", e.label], ["value", e.secure ? "(secure)" : e.value],
        ["frame", e.frame && e.frame.map(Math.round).join(", ")], ["enabled", e.enabled], ["focused", e.focused],
        ["actions", (e.actions || []).join(", ")], ["interactive", e.interactive], ["secure", e.secure]];
      for (const [k, v] of rows) {
        const d = el("div");
        d.appendChild(el("b", null, k + ": "));
        d.appendChild(document.createTextNode(v == null ? "" : String(v)));
        a.appendChild(d);
      }
    }
    const can = (act) => !!e && e.enabled && !e.secure && (e.actions || []).includes(act);
    $("act-press").disabled = !can("press");
    $("act-focus").disabled = !e || e.secure || !e.enabled;
    $("act-menu").disabled = !can("show_menu");
    $("scrtypeform").hidden = !e || e.secure || !TEXTY.has(e.role);
  }
  function act(action, value) {
    if (!sel) return;
    const m = { type: "screen_action", id: sel, action };
    if (value != null) m.value = value;
    send(m);
  }
  $("act-press").onclick = () => act("press");
  $("act-focus").onclick = () => act("focus");
  $("act-menu").onclick = () => act("show_menu");
  $("scrtypeform").addEventListener("submit", (ev) => {
    ev.preventDefault();
    act("set_value", $("scrvalue").value);
  });
  function scan() { send({ type: "screen_snapshot", app: $("scrapp").value.trim() || null }); }
  $("scrscan").onclick = scan;
  $("scrapp").addEventListener("keydown", (ev) => { if (ev.key === "Enter") scan(); });
  function onScreenResult(m) {
    const s = $("scrstatus");
    s.textContent = `${m.action} ${m.id}: ${m.result}`;
    s.classList.toggle("bad", !m.ok);
    if (!m.ok) toast(m.result);
    else { const t = el("div", "toast", m.result); t.style.cssText = "border-color:var(--fg);color:var(--fg);background:var(--panel)"; $("toasts").appendChild(t); setTimeout(() => t.remove(), 4000); }
  }

  // ---- composer ----
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("input").value.trim();
    if (!text && !attached) return;
    if (send({ type: "text", text, image: attached, use_camera: useCam })) {
      $("input").value = "";
      setAttached(null);
    }
  });
  $("stop").onclick = () => send({ type: "stop" });
  $("cambtn").onclick = () => {
    useCam = !useCam;
    $("cambtn").setAttribute("aria-pressed", useCam);
  };
  $("attachbtn").onclick = () => $("file").click();
  $("file").onchange = (e) => { if (e.target.files[0]) attachFile(e.target.files[0]); e.target.value = ""; };
  $("attachrm").onclick = () => setAttached(null);

  function setAttached(b64) {
    attached = b64;
    $("attachrow").hidden = !b64;
    if (b64) $("attachimg").src = "data:image/jpeg;base64," + b64;
  }
  function attachFile(f) {
    if (!f.type.startsWith("image/")) return toast("Not an image file");
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, 1024 / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      setAttached(c.toDataURL("image/jpeg", 0.85).split(",")[1]);
    };
    img.onerror = () => { URL.revokeObjectURL(url); toast("Could not read image"); };
    img.src = url;
  }
  const firstImage = (files) => Array.from(files || []).find((f) => f.type.startsWith("image/"));
  let dragDepth = 0;
  addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("drop").hidden = false; });
  addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; } });
  addEventListener("dragover", (e) => e.preventDefault());
  addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; $("drop").hidden = true;
    const f = firstImage(e.dataTransfer.files); if (f) attachFile(f);
  });
  addEventListener("paste", (e) => {
    const f = firstImage(e.clipboardData && e.clipboardData.files); if (f) attachFile(f);
  });

  // ---- camera feed with retry ----
  const vid = $("video");
  vid.onload = () => { $("camoff").hidden = true; };
  vid.onerror = () => {
    $("camoff").hidden = false;
    setTimeout(() => { vid.src = "/video.mjpg?t=" + Date.now(); }, 4000);
  };

  // ---- faint falling code: ~12 fps, off for reduced motion ----
  const rainOn = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (rainOn) {
    const cv = $("rain"), g = cv.getContext("2d"), SZ = 16;
    const glyphs = "01ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿ<>{}[]=+*";
    let drops = [], last = 0;
    const fit = () => {
      cv.width = innerWidth; cv.height = innerHeight;
      drops = Array.from({ length: Math.ceil(cv.width / SZ) }, () => Math.random() * -50);
    };
    addEventListener("resize", fit); fit();
    const tick = (now) => {
      requestAnimationFrame(tick);
      if (now - last < 80) return;
      last = now;
      g.fillStyle = "rgba(2,10,4,.12)"; g.fillRect(0, 0, cv.width, cv.height);
      g.fillStyle = "#00ff41"; g.font = SZ + "px monospace";
      drops.forEach((y, i) => {
        g.fillText(glyphs[Math.floor(Math.random() * glyphs.length)], i * SZ, y * SZ);
        drops[i] = y * SZ > cv.height && Math.random() > 0.975 ? 0 : y + 1;
      });
    };
    requestAnimationFrame(tick);
  }

  setState("idle");
  connect();
})();
