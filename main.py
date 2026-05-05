import argparse
import faulthandler
import re
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

from src.processor import DemoProcessor
from src.utils import CSGOUtils


class PipelineRunner:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        tick_step: int = 64,
        timeout_seconds: int = 600,
        max_workers: int = 1,
        max_retries: int = 1,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tick_step = tick_step
        self.timeout_seconds = timeout_seconds
        self.max_workers = max(1, int(max_workers))
        self.max_retries = max(0, int(max_retries))
        self.failed_log_path = self.output_dir / "failed_demos.log"

    @staticmethod
    def _safe_series_name(raw_series_name: str) -> str:
        series_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_series_name.strip()).strip("_").lower()
        return series_name or "series"

    @staticmethod
    def _demo_base_name(demo_file: Path) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", demo_file.stem.strip()).strip("_").lower()
        return base_name or "match"

    @staticmethod
    def _expected_csv_path(demo_file: Path, out_series_dir: Path) -> Path:
        return out_series_dir / f"{PipelineRunner._demo_base_name(demo_file)}.csv"

    @staticmethod
    def _process_demo_worker(
        demo_path: str,
        series_name: str,
        output_series_dir: str,
        tick_step: int,
        result_queue,
    ) -> None:
        # Ha natív library crash-el, a stack trace legalább dumpolódik.
        faulthandler.enable(all_threads=True)

        demo_file = Path(demo_path)
        out_series_dir = Path(output_series_dir)

        try:
            processor = DemoProcessor(str(demo_file), tick_step=tick_step)
            df = processor.process()

            if df.height <= 0:
                result_queue.put({"status": "empty", "demo_name": demo_file.name})
                return

            # Determinisztikus kimeneti név: így a futás folytatható skip-pel.
            save_path = PipelineRunner._expected_csv_path(demo_file, out_series_dir)

            file_name = save_path.name

            df.write_csv(str(save_path))
            result_queue.put(
                {
                    "status": "ok",
                    "demo_name": demo_file.name,
                    "file_name": file_name,
                }
            )
        except Exception as e:
            result_queue.put(
                {
                    "status": "error",
                    "demo_name": demo_file.name,
                    "error": str(e),
                }
            )

    @staticmethod
    def _is_likely_native_crash_exit(exit_code: int) -> bool:
        # Linux/macOS: SIGSEGV -> -11
        # Windows:
        #   0xC0000005 = access violation
        #   0xC0000374 = heap corruption
        #   0xC00000FD = stack overflow
        if exit_code == -11:
            return True
        return (exit_code & 0xFFFFFFFF) in {0xC0000005, 0xC0000374, 0xC00000FD}

    def _record_failure(self, demo_file: Path, series_name: str, reason: str, attempt: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}\tseries={series_name}\tdemo={demo_file.name}\tattempt={attempt}\treason={reason}\n"
        with self.failed_log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _remove_failure_entries(self, demo_file: Path, series_name: str) -> None:
        if not self.failed_log_path.exists():
            return
        series_token = f"series={series_name}"
        demo_token = f"demo={demo_file.name}"
        with self.failed_log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        kept = [ln for ln in lines if not (series_token in ln and demo_token in ln)]
        if len(kept) == len(lines):
            return
        if kept:
            with self.failed_log_path.open("w", encoding="utf-8") as f:
                f.writelines(kept)
        else:
            self.failed_log_path.unlink(missing_ok=True)

    def _start_demo_process(self, demo_file: Path, series_name: str, out_series: Path, attempt: int):
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=PipelineRunner._process_demo_worker,
            args=(
                str(demo_file),
                series_name,
                str(out_series),
                self.tick_step,
                result_queue,
            ),
        )
        process.start()
        return {
            "process": process,
            "result_queue": result_queue,
            "demo_file": demo_file,
            "series_name": series_name,
            "out_series": out_series,
            "attempt": attempt,
            "started_at": time.monotonic(),
        }

    def _finalize_demo_process(self, job):
        process = job["process"]
        result_queue = job["result_queue"]
        demo_file = job["demo_file"]
        series_name = job["series_name"]

        if process.is_alive():
            process.terminate()
            process.join()
            print(f" IDŐTÚLLÉPÉS: {demo_file.name} ({self.timeout_seconds} mp)")
            return "timeout", f"timeout_{self.timeout_seconds}s"

        if process.exitcode != 0:
            if self._is_likely_native_crash_exit(process.exitcode):
                print(f" CRASH/NATIVE: {demo_file.name} (exit code: {process.exitcode})")
                return "native_crash", f"native_crash_exit_{process.exitcode}"
            else:
                print(f" HIBA: {demo_file.name} subprocess leállt (exit code: {process.exitcode})")
                return "subprocess_error", f"subprocess_exit_{process.exitcode}"

        try:
            result = result_queue.get_nowait()
        except Empty:
            print(f" HIBA: {demo_file.name} nem küldött eredményt.")
            return "no_result", "no_result_from_worker"

        status = result.get("status")
        if status == "ok":
            print(f" SIKER: {result['demo_name']} -> {result['file_name']}")
            return "ok", ""
        elif status == "empty":
            print(f" ÜRES: {result['demo_name']} (nem generálódott adat)")
            return "empty", ""
        else:
            print(
                f" HIBA: {result.get('demo_name', demo_file.name)} feldolgozása sikertelen: "
                f"{result.get('error', 'ismeretlen hiba')}"
            )
            return "error", f"python_error_{result.get('error', 'ismeretlen_hiba')}"

    def run(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input mappa nem található: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for tournament_folder in self.input_dir.iterdir():
            if not tournament_folder.is_dir():
                continue

            print(f"\n=== TOURNAMENT: {tournament_folder.name} ===")
            out_tournament = self.output_dir / tournament_folder.name
            out_tournament.mkdir(parents=True, exist_ok=True)

            for series_folder in tournament_folder.iterdir():
                if not series_folder.is_dir():
                    continue

                print(f"\n--- SERIES: {series_folder.name} ---")
                out_series = out_tournament / series_folder.name
                out_series.mkdir(parents=True, exist_ok=True)

                pending = []
                for demo_file in series_folder.glob("*.dem"):
                    expected_csv = self._expected_csv_path(demo_file, out_series)
                    if expected_csv.exists():
                        print(f" SKIP (már kész): {demo_file.name} -> {expected_csv.name}")
                        self._remove_failure_entries(demo_file=demo_file, series_name=series_folder.name)
                        continue
                    pending.append({"demo_file": demo_file, "attempt": 1})
                active = []

                while pending or active:
                    while pending and len(active) < self.max_workers:
                        item = pending.pop(0)
                        demo_file = item["demo_file"]
                        attempt = item["attempt"]
                        active.append(
                            self._start_demo_process(
                                demo_file=demo_file,
                                series_name=series_folder.name,
                                out_series=out_series,
                                attempt=attempt,
                            )
                        )

                    finished_jobs = []
                    for job in active:
                        process = job["process"]
                        elapsed = time.monotonic() - job["started_at"]

                        if process.is_alive() and elapsed <= self.timeout_seconds:
                            continue

                        if process.is_alive() and elapsed > self.timeout_seconds:
                            process.terminate()
                            process.join()

                        result_code, result_reason = self._finalize_demo_process(job)
                        if result_code in {"native_crash", "subprocess_error", "no_result"} and job["attempt"] <= self.max_retries:
                            next_attempt = job["attempt"] + 1
                            print(f" ÚJRAPRÓBA: {job['demo_file'].name} (attempt {next_attempt}/{self.max_retries + 1})")
                            pending.insert(0, {"demo_file": job["demo_file"], "attempt": next_attempt})
                        elif result_code in {"ok", "empty"}:
                            self._remove_failure_entries(
                                demo_file=job["demo_file"],
                                series_name=job["series_name"],
                            )
                        elif result_code not in {"ok", "empty"}:
                            # Csak végleg elbukott fájl kerüljön a logba.
                            self._record_failure(
                                demo_file=job["demo_file"],
                                series_name=job["series_name"],
                                reason=result_reason,
                                attempt=job["attempt"],
                            )
                        finished_jobs.append(job)

                    for job in finished_jobs:
                        active.remove(job)

                    if active:
                        time.sleep(0.10)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CS demo -> CSV pipeline (Windows-barát, izolált feldolgozással)"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Gyökér mappa, ahol a tournament/series/.dem struktúra található",
    )
    parser.add_argument(
        "--output-dir",
        default="./processed",
        help="Kimeneti mappa a CSV fájlokhoz",
    )
    parser.add_argument(
        "--tick-step",
        type=int,
        default=16,
        help="Mintavételezés tick lépésköze",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Max feldolgozási idő / demo (mp)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Párhuzamosan futó demo processzek száma (1 = stabil alapmód)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Automatikus újrapróba natív crash/subprocess hiba esetén",
    )
    return parser.parse_args()


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    args = parse_args()

    runner = PipelineRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tick_step=args.tick_step,
        timeout_seconds=args.timeout_seconds,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    runner.run()
    print("\nFeldolgozás befejezve.")
