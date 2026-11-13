"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), orb = $("orb");
  let ws, retry = 0, stick = true, attached = null, useCam = false;
  const bubbles = new Map(); // turn_id -> assistant bubble element

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const clock = (ts) => new Date((ts || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  // ---- chat rendering ----
  chat.addEventListener("scroll", () => {
    stick = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 40;
  });
  const scroll = () => { if (stick) chat.scrollTop = chat.scrollHeight; };

  function bubble(role, text, ts, extra) {
    const b = el("div", "msg " + role);
    if (extra && extra.image) {
      const im = el("img");
      im.alt = "Attached photo";
      im.src = extra.image; // data: URL from server, set as attribute value (not HTML)
      b.appendChild(im);
    }
    b.appendChild(el("span", "body", (extra && extra.icon ? extra.icon + " " : "") + text));
    b.appendChild(el("span", "meta", clock(ts)));
    chat.appendChild(b);
    scroll();
    return b;
  }
  const setBody = (b, t, icon) => { b.querySelector(".body").textContent = (icon || "") + t; };

  function renderHistory(h) {
    chat.textContent = "";
    bubbles.clear();
    for (const m of h || []) bubble(m.role === "user" ? "user" : "assistant", m.text, m.ts);
    stick = true; scroll();
  }

  function onUser(m) {
    bubble("user", m.text, m.ts, { image: m.image, icon: m.source === "voice" ? "\u{1F3A4}" : "⌨️" });
  }
  function onDelta(m) {
    let b = bubbles.get(m.turn_id);
    if (!b) {
      b = bubble("assistant", "", null);
      b.classList.add("streaming");
      bubbles.set(m.turn_id, b);
    }
    const body = b.querySelector(".body");
    body.textContent += m.text;
    scroll();
  }
  function onAssistant(m) {
    let b = bubbles.get(m.turn_id);
    if (!b) b = bubble("assistant", "", m.ts);
    bubbles.delete(m.turn_id);
    b.classList.remove("streaming");
    if (m.outcome === "cancelled" || m.outcome === "failed") b.classList.add(m.outcome);
    // final text replaces streamed text; keep streamed text if final is empty (e.g. cancelled)
    if (m.text) setBody(b, m.text);
    const meta = b.querySelector(".meta");
    const parts = [clock(m.ts)];
    if (m.latency_ms != null) parts.push(Math.round(m.latency_ms) + " ms");
    if (m.outcome && m.outcome !== "completed") parts.push(m.outcome);
    meta.textContent = parts.join(" · ");
    scroll();
  }

  // ---- status / hello / vision / errors ----
  function setState(s) {
    orb.dataset.state = s;
    orb.setAttribute("aria-label", "Assistant state: " + s);
    $("statelabel").textContent = s.toUpperCase();
  }
  function flag(id, label, on) {
    const e = $(id);
    e.textContent = label + " " + (on ? "on" : "off");
    e.className = "chip " + (on ? "on" : "");
  }
  function onVision(m) {
    const c = $("caption");
    c.textContent = m.text; c.hidden = false;
    const li = el("li");
    li.appendChild(el("time", null, clock(m.ts)));
    li.appendChild(document.createTextNode(m.text));
    const ul = $("vlist");
    ul.insertBefore(li, ul.firstChild);
    while (ul.children.length > 50) ul.removeChild(ul.lastChild);
    $("vcount").textContent = ul.children.length;
  }
  function toast(msg) {
    const t = el("div", "toast");
    t.appendChild(el("span", null, msg));
    const x = el("button", null, "×");
    x.type = "button"; x.setAttribute("aria-label", "Dismiss");
    x.onclick = () => t.remove();
    t.appendChild(x);
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 10000);
  }
  function conn(on) {
    const c = $("conn");
    c.textContent = on ? "ONLINE" : "OFFLINE";
    c.className = "chip " + (on ? "on" : "off");
  }

  function handle(m) {
    switch (m.type) {
      case "hello":
        $("model").textContent = "model " + (m.model || "—");
        flag("camflag", "cam", m.camera); flag("visflag", "vision", m.vision);
        renderHistory(m.history);
        break;
      case "status": setState(m.state); break;
      case "user": onUser(m); break;
      case "assistant_delta": onDelta(m); break;
      case "assistant": onAssistant(m); break;
      case "vision": onVision(m); break;
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
  const vid = $("video"), cam = $("cam");
  vid.onload = () => { cam.classList.remove("dead"); $("camoff").hidden = true; };
  vid.onerror = () => {
    cam.classList.add("dead"); $("camoff").hidden = false;
    setTimeout(() => { vid.src = "/video.mjpg?t=" + Date.now(); }, 4000);
  };

  setState("idle");
  connect();
})();
