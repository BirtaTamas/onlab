import os
import re
from pathlib import Path
from datetime import datetime
from src.processor import DemoProcessor
from src.utils import CSGOUtils

class PipelineRunner:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def run(self):
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

                for demo_file in series_folder.glob("*.dem"):
                    # Időbélyeg generálása a névhez
                    timestamp = datetime.now().strftime("%H%M%S")

                    try:
                        processor = DemoProcessor(str(demo_file))
                        df = processor.process()

                        if df.height > 0:
                            map_name = CSGOUtils.infer_map_name(demo_file.name, series_folder.name)
                            series_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", series_folder.name.strip()).strip("_").lower()
                            if not series_name:
                                series_name = "series"

                            file_name = f"{series_name}_{map_name}_{timestamp}.csv"
                            save_path = out_series / file_name

                            df.write_csv(str(save_path))
                            print(f" SIKER: {demo_file.name} -> {file_name}")
                        else:
                            print(f" ÜRES: {demo_file.name} (nem generálódott adat)")

                    except Exception as e:
                        print(f" HIBA: {demo_file.name} feldolgozása sikertelen: {e}")

if __name__ == "__main__":
    # A te elérési utad
    INPUT_PATH = "/Users/birtatamas/Documents/cs"
    OUTPUT_PATH = "./data/processed"
    
    runner = PipelineRunner(INPUT_PATH, OUTPUT_PATH)
    runner.run()
    print("\nFeldolgozás befejezve.")
