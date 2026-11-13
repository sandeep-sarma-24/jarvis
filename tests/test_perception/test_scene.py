"""Scene tracker, VLM gate (fake clock), detector disabled path, crop. No model, no camera."""
from __future__ import annotations

import sys

from app.config import DetectorConfig
from app.perception.detector import Detector
from app.perception.scene import SceneTracker, SceneWatcher, VlmGate, crop_frame
from app.perception.types import Entity, Frame


def ent(i, label="mug", conf=0.8):
    return Entity(i, label, conf, (0.1, 0.1, 0.3, 0.3))


def feed(tr, t0, t1, ents, step=0.2):
    """Feed the same entities every `step` seconds; returns all events."""
    out, t = [], t0
    while t <= t1 + 1e-9:
        out += tr.update(t, ents)
        t += step
    return out


def test_enter_needs_frames_and_age():
    tr = SceneTracker()
    assert tr.update(0.0, [ent(1)]) == [] and tr.update(0.2, [ent(1)]) == [] and tr.update(0.4, [ent(1)]) == []
    assert tr.counts() == {}  # 3 frames but age < 1s
    ev = tr.update(1.0, [ent(1)])
    assert [str(e) for e in ev] == ["mug entered"] and ev[0].new_class
    assert tr.counts() == {"mug": 1}


def test_person_confirms_after_two_frames_and_low_conf_resets():
    tr = SceneTracker()
    tr.update(0.0, [ent(1, "person")])
    assert [str(e) for e in tr.update(0.2, [ent(1, "person")])] == ["person entered"]
    tr2 = SceneTracker()
    tr2.update(0, [ent(1)]), tr2.update(0.5, [ent(1)]), tr2.update(0.9, [ent(1, conf=0.3)])
    assert tr2.update(1.2, [ent(1)]) == []  # streak was reset


def test_left_after_one_second_and_unconfirmed_leaves_silently():
    tr = SceneTracker()
    feed(tr, 0, 1.2, [ent(1)])
    assert tr.update(2.0, []) == []          # missing 0.8s: still there
    assert tr.counts() == {"mug": 1}
    assert [str(e) for e in tr.update(2.3, [])] == ["mug left"]
    assert tr.counts() == {}
    tr.update(3.0, [ent(2)])
    assert tr.update(5.0, []) == []          # never confirmed


def test_new_class_only_after_memory_window():
    tr = SceneTracker()
    feed(tr, 0, 1.2, [ent(1)])
    ev = feed(tr, 1.4, 3.0, [ent(1), ent(2)])
    assert [e.new_class for e in ev if e.kind == "entered"] == [False]  # second mug: class already present
    feed(tr, 3.2, 5.0, [])
    ev = feed(tr, 5.2, 7.0, [ent(3)])
    assert not ev[0].new_class               # mug seen < 60s ago
    feed(tr, 7.2, 9.0, [])
    ev = feed(tr, 80.0, 82.0, [ent(4)])
    assert ev[0].new_class


def test_counts_and_summary():
    tr = SceneTracker()
    ents = [ent(1, "person"), ent(2, "laptop"), ent(3, "mug"), ent(4, "mug")]
    feed(tr, 0, 1.2, ents)
    assert tr.counts() == {"person": 1, "laptop": 1, "mug": 2}
    s = tr.summary(3.0)
    assert s.startswith("mug×2, laptop×1, person×1 (") and "entered" in s and "s ago)" in s
    assert SceneTracker().summary(0) == "nothing detected"
    assert "(" not in tr.summary(100.0)      # recent events expired


def make(person=True):
    tr, gate = SceneTracker(), VlmGate()
    if person:
        feed(tr, 0, 0.4, [ent(1, "person")])
    return tr, gate


def test_gate_new_class_fires_then_cooldown_drops():
    tr, gate = SceneTracker(), VlmGate()
    ev = feed(tr, 0, 0.4, [ent(1, "person")])
    req = gate.step(10.0, tr, ev)
    assert req.kind == "new_class" and req.track_ids == (1,)
    ev = feed(tr, 10.2, 11.4, [ent(1, "person"), ent(2)])  # mug confirmed ~1s later, inside... cooldown is 3s
    assert gate.step(11.5, tr, ev) is None and gate.dropped == 1
    # the dropped trigger recovers as a stable change once the cooldown is over
    assert gate.step(14.0, tr, []).kind in ("scene_change", "count_change")


def test_gate_busy_drops_and_recovers():
    tr, gate = make()
    ev = [e for e in feed(SceneTracker(), 0, 0.4, [ent(1, "person")])]
    assert gate.step(10.0, tr, ev, busy=True) is None
    assert gate.step(10.5, tr, [], busy=False).kind == "scene_change"  # level-triggered: ids still differ


def test_gate_count_change_alone():
    tr, gate = SceneTracker(), VlmGate(jaccard=0.99)
    ents = [ent(i, "mug") for i in range(1, 4)]
    feed(tr, 0, 1.2, ents)
    gate.step(10.0, tr, [])                     # baseline (new-class not passed; ids differ but jaccard off)
    tr.update(10.0, ents[:2])
    gate.step(10.1, tr, [])
    tr.update(12.5, ents[:2])                   # id 3 gone > left_after
    assert tr.counts() == {"mug": 2}
    assert gate.step(12.6, tr, []) is None      # changed 0.1s ago
    assert gate.step(15.0, tr, []).kind == "count_change"


def test_gate_rate_limit_and_refresh_timers():
    tr = SceneTracker()
    gate = VlmGate(cooldown=0, max_per_min=3, refresh_person=60, refresh_empty=180, start=0.0)
    assert gate.step(59.0, tr, []) is None and gate.step(179.0, tr, []) is None
    assert gate.step(180.0, tr, []).kind == "refresh"         # empty room
    gate2 = VlmGate(cooldown=0, max_per_min=2, start=0.0)
    tr2 = SceneTracker()
    ev = feed(tr2, 0, 0.4, [ent(1, "person")])
    got = [gate2.step(10 + i, tr2, ev) for i in range(3)]
    assert got[0] is not None and got[1] is not None and got[2] is None  # 2/min cap


def test_gate_person_refresh_is_60s():
    tr = SceneTracker()
    feed(tr, 0, 0.4, [ent(1, "person")])
    gate = VlmGate(start=0.0)
    gate._base_ids, gate._base_counts, gate._pend_counts = tr.ids(), tr.counts(), tr.counts()
    assert gate.step(59.0, tr, []) is None
    assert gate.step(60.0, tr, []).kind == "refresh"


def test_gate_sustained_motion():
    tr, gate = SceneTracker(), VlmGate(start=0.0)
    assert gate.step(10.0, tr, [], frame_changed=True) is None
    assert gate.step(10.5, tr, [], frame_changed=True) is None
    assert gate.step(11.0, tr, [], frame_changed=True).kind == "scene_change"


def test_detector_disabled_without_ultralytics(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import raises ImportError
    d = Detector(DetectorConfig())
    assert d.load() is False and not d.loaded
    assert "Detector disabled" in caplog.text
    w = SceneWatcher(detector=d, camera=None, change=None)
    assert not w.available and w.summary() is None


def test_crop_frame_pads_and_clamps():
    data = bytes(range(256)) * 3  # 16x16x3
    f = Frame(0.0, data, 16, 16, "rgb24")
    c = crop_frame(f, (0.25, 0.25, 0.5, 0.5), pad=0.2)
    assert (c.width, c.height) == (6, 6) and len(c.data) == 6 * 6 * 3
    edge = crop_frame(f, (0.0, 0.0, 0.1, 0.1))
    assert edge.width >= 1 and edge.height >= 1


async def test_watcher_calls_describe_with_hint_and_respects_idle():
    import asyncio
    from app.perception.camera import FakeCamera
    from app.perception.visual import ChangeDetector

    class FakeDetector:
        loaded = True
        def detect(self, frame):
            return [Entity(1, "person", 0.9, (0.0, 0.0, 0.5, 0.5))]

    clock = [0.0]
    cam = FakeCamera(width=8, height=8)
    await cam.start()
    calls, updates = [], []
    idle = asyncio.Event()  # a turn is running
    w = SceneWatcher(detector=FakeDetector(), camera=cam, change=ChangeDetector(change_threshold=2.0),
                     idle=idle, on_update=updates.append, clock=lambda: clock[0])

    async def describe(frame, hint):
        calls.append((frame, hint))
    w.describe = describe
    loop = asyncio.get_running_loop()
    for _ in range(3):
        clock[0] += 0.2
        await w._tick(loop, await cam.capture())
    await asyncio.sleep(0)
    assert calls == [] and w.summary() is not None  # busy turn: dropped
    assert updates[-1]["items"][0]["label"] == "person" and "ms" in updates[-1]
    idle.set()
    clock[0] += 4.0  # past the cooldown measured from startup
    await w._tick(loop, await cam.capture())
    await asyncio.sleep(0.05)
    assert len(calls) == 1 and "person" in calls[0][1] and calls[0][0].width < 8  # crop for a single entity
    clock[0] += 30
    assert w.summary() is None                      # stale detector: no summary
    w._pool.shutdown()
