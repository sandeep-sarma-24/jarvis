"""Pure helpers of the local-stack scripts: no models, no network, no devices."""
from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from scripts import benchmark_local_stack as bench
from scripts import test_local_e2e as e2e

ROOT = Path(__file__).resolve().parents[2]


def test_parse_swapusage() -> None:
    s = bench.parse_swapusage("total = 4096.00M  used = 3111.56M  free = 984.44M  (encrypted)")
    assert s == {"total_mb": 4096.0, "used_mb": 3111.56, "free_mb": 984.44}
    assert bench.parse_swapusage("total = 2.00G  used = 512.00K  free = 2.00G")["total_mb"] == 2048.0
    assert bench.parse_swapusage("garbage") is None


def test_parse_memory_pressure() -> None:
    out = "The system has 17179869184 (1048576 pages)\nPageouts: 1\n\nSystem-wide memory free percentage: 72%\n"
    assert bench.parse_memory_pressure(out) == 72
    assert bench.parse_memory_pressure("") is None


def test_parse_thermal() -> None:
    nominal = ("Note: No thermal warning level has been recorded\n"
               "Note: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n")
    assert bench.parse_thermal(nominal) == "nominal"
    hot = "CPU_Scheduler_Limit \t= 100\nCPU_Speed_Limit \t= 70\nThermal Warning Level = 2\n"
    assert "CPU_Speed_Limit" in bench.parse_thermal(hot) and "Thermal Warning Level" in bench.parse_thermal(hot)


def test_parse_ollama_ps() -> None:
    data = {"models": [{"name": "gemma4:e2b", "size": 4 * 2**30, "size_vram": 3 * 2**30, "context_length": 4096}]}
    assert bench.parse_ollama_ps(data) == [{"name": "gemma4:e2b", "size_mb": 4096.0, "size_vram_mb": 3072.0,
                                            "gpu_fraction": 0.75, "context_length": 4096}]
    assert bench.parse_ollama_ps({"models": []}) == []
    assert bench.parse_ollama_ps({"models": [{"model": "x", "size": 0}]})[0]["gpu_fraction"] is None


def test_select_scenarios() -> None:
    assert bench.select_scenarios(None) == "ABCDE"
    assert bench.select_scenarios("all") == "ABCDEF"
    assert bench.select_scenarios("F") == "F"
    assert bench.select_scenarios("e,a") == "AE"
    for bad in ("G", "", " , "):
        if bad:
            with pytest.raises(ValueError):
                bench.select_scenarios(bad)
    with pytest.raises(SystemExit):
        bench.parse_args(["--scenario", "Z"])


def test_summarize_rows_counts_failures_and_overlap() -> None:
    rows = [{"ttft_ms": 100.0, "side_asr_overlapped_llm": True}, {"ttft_ms": 200.0, "side_asr_overlapped_llm": False},
            {"error": "boom"}]
    s = bench.summarize_rows(rows)
    assert s["ttft_ms"]["n"] == 2 and s["failures"] == ["boom"] and s["side_asr_overlap_confirmed"] == "1/2"
    assert bench.fmt(s["asr_ms"]) == bench.NOT_RUN


@pytest.mark.parametrize("address, local", [
    (("127.0.0.1", 11434), True), (("::1", 11434, 0, 0), True), (("localhost", 80), True),
    ("/tmp/sock", True), (("10.0.0.5", 11434), False), (("100.101.102.103", 11434), False),
    (("huggingface.co", 443), False), (("2001:db8::1", 443, 0, 0), False),
])
def test_is_local_destination(address, local) -> None:
    assert e2e.is_local_destination(address) is local


def test_connection_recorder_records_and_flags_remote() -> None:
    rec = e2e.ConnectionRecorder()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    addr = srv.getsockname()
    rec.install()
    try:
        for method in ("connect", "connect_ex"):
            with socket.socket() as c:
                getattr(c, method)(addr)
    finally:
        rec.uninstall()
        srv.close()
    assert [a for _, a in rec.destinations] == [addr, addr] and rec.remote == []
    assert "wrapper" not in socket.socket.connect.__name__  # uninstalled
    rec.destinations.append((socket.AF_INET, ("192.0.2.1", 443)))  # no real remote connect in a unit test
    assert rec.remote == [("192.0.2.1", 443)]


def test_e2e_script_is_not_a_pytest_module() -> None:
    assert not [n for n in vars(e2e) if n.startswith("test")]


@pytest.mark.parametrize("script", ["check_local_environment.sh", "setup_local_models.sh"])
def test_shell_script_syntax(script) -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)


def test_parse_footprint_sums_processes() -> None:
    out = ("======\nollama [26769]: 64-bit    Footprint: 19 MB (16384 bytes per page)\n======\n"
           "ollama [32352]: 64-bit    Footprint: 7.03 GB (16384 bytes per page)\n    phys_footprint: 7.03 GB\n"
           "Summary Footprint: 7218 MB\n")
    assert bench.parse_footprint(out) == round(19 + 7.03 * 1024, 1)
    assert bench.parse_footprint("footprint: no such process") is None
