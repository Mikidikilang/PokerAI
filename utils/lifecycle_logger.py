import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

class LifeCycleLogger:
    """
    A PokerAI modell életút-naplózó rendszere.
    Egyetlen monolitikus JSON fájlban tárolja a betanítási és validációs
    metrikákat, valamint a hiperparaméter-konfigurációkat epizódokra/mérföldkövekre bontva.
    """
    def __init__(self, model_id: str, log_dir: str = "logs/lifecycle"):
        self.model_id = model_id
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / f"{model_id}_lifecycle.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Betölti a meglévő naplót, vagy létrehoz egy új, üres struktúrát."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Figyelem: A naplófájl ({self.log_file}) sérült. Új struktúra inicializálása.")
        
        # Alap struktúra a specifikáció alapján
        return {
            "model_id": self.model_id,
            "creation_timestamp": datetime.utcnow().isoformat() + "Z",
            "episodes": []
        }

    def _save_atomic(self):
        """Atomi fájlírás a korrupció elkerülése végett (pl. váratlan leállás esetén)."""
        temp_fd, temp_path = tempfile.mkstemp(dir=self.log_dir, suffix='.json')
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            
            # Fájl felülírása atomi módon
            shutil.move(temp_path, self.log_file)
        except Exception as e:
            os.remove(temp_path)
            raise e

    def log_milestone(self, 
                      episode_id: int, 
                      config: dict, 
                      train_metrics: dict, 
                      validation_metrics: dict):
        """
        Rögzít egy új mérföldkövet a betanítási folyamatban.
        
        :param episode_id: Az aktuális lépésszám vagy epizód azonosító.
        :param config: Az éppen használt konfiguráció (hiperparaméterek, PPO beállítások, stb).
        :param train_metrics: A betanítás alatti mutatók (policy_loss, value_loss, entropy).
        :param validation_metrics: A játékelméleti metrikák (VPIP, PFR, winrate).
        """
        
        episode_data = {
            "episode_id": episode_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "configuration": config,
            "training_metrics": train_metrics,
            "validation_metrics": validation_metrics
        }
        
        # Frissítjük a memóriában lévő adatot és kiírjuk lemezre
        self.data["episodes"].append(episode_data)
        self._save_atomic()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Életút-napló frissítve. Modell: {self.model_id} | Epizód: {episode_id}")

    def get_dataframe(self):
        """
        Opcionális segédfüggvény Pandas integrációhoz.
        Kisimítja (flatten) a JSON struktúrát DataFrame létrehozásához.
        """
        try:
            import pandas as pd
            
            flattened_data = []
            for ep in self.data["episodes"]:
                row = {"episode_id": ep["episode_id"], "timestamp": ep["timestamp"]}
                
                # Konfigurációk prefixálása
                for k, v in ep.get("configuration", {}).items():
                    row[f"cfg_{k}"] = v
                    
                # Train metrikák prefixálása
                for k, v in ep.get("training_metrics", {}).items():
                    row[f"train_{k}"] = v
                    
                # Validációs metrikák prefixálása
                for k, v in ep.get("validation_metrics", {}).items():
                    row[f"val_{k}"] = v
                    
                flattened_data.append(row)
                
            return pd.DataFrame(flattened_data)
        except ImportError:
            print("A Pandas könyvtár nem található. Telepítés: pip install pandas")
            return None