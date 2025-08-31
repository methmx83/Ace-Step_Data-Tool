"""
scripts/helpers/logger_setup.py
Zentrales Logging-System für das Audio-Tagging-Projekt

Features:
- Dual-Output: Terminal + Log-Datei  
- Timestamp-basierte Log-Dateien
- Verschiedene Log-Level für verschiedene Ausgaben
- Strukturierte Logs für bessere Analyse
- Automatische Log-Rotation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os
import re

# Globale Session-Infos für Zusatz-Logs (z. B. RAW Responses)
_CURRENT_SESSION_NAME: Optional[str] = None
_RAW_BASE_DIR = Path("logs") / "raw"
# Wenn True, unterdrücke ausführliche Header-Logs (System-Info, Config Dumps, SessionStart)
_SUPPRESS_HEADER: bool = False

class DualLogger:
    """
    Logger der gleichzeitig ins Terminal und in Datei schreibt
    """
    
    @staticmethod
    def setup_logging(
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        session_name: Optional[str] = None
    ) -> logging.Logger:
        """
        Richtet Dual-Logging ein
        
        Args:
            log_dir: Verzeichnis für Log-Dateien
            log_level: Standard Log-Level
            console_level: Level für Terminal-Output
            file_level: Level für Datei-Output  
            session_name: Name für diese Session (optional)
            
        Returns:
            Konfigurierter Logger
        """
        
        # Sicherstellen dass Log-Directory existiert
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(exist_ok=True)
        
    # Session-Name generieren falls nicht gegeben
        if session_name is None:
            session_name = f"tagging_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log-Datei-Pfad
        log_file = log_dir_path / f"{session_name}.log"
        
    # Root Logger konfigurieren
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Alle vorhandenen Handler entfernen
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Formatter definieren
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler (Terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File Handler (Log-Datei)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # RAW-Ordner für diese Session vorbereiten
        try:
            global _CURRENT_SESSION_NAME
            _CURRENT_SESSION_NAME = session_name
            (_RAW_BASE_DIR / session_name).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # RAW Directory ist optional, daher nur debug-loggen
            tmp_logger = logging.getLogger("LoggerSetup")
            tmp_logger.debug(f"Could not prepare RAW directory: {e}")

        # Session-Info loggen
        # Nur loggen wenn Header nicht unterdrückt wird
        if not _SUPPRESS_HEADER:
            logger = logging.getLogger("LoggerSetup")
            logger.info(f"Logging session started: {session_name}")
            logger.info(f"Log file: {log_file}")
            logger.info(f"Console level: {logging.getLevelName(console_level)}")
            logger.info(f"File level: {logging.getLevelName(file_level)}")
        
        return root_logger
    
    @staticmethod
    def log_system_info():
        """Loggt System-Informationen für besseres Debugging"""
        import platform
        import torch
        
        # Respektiere globale Suppress-Einstellung
        if _SUPPRESS_HEADER:
            return

        logger = logging.getLogger("SystemInfo")

        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"🖥️ Platform: {platform.platform()}")
        logger.info(f"🐍 Python: {platform.python_version()}")
        logger.info(f"🧩 PyTorch: {torch.__version__}")
        logger.info(f"ℹ️ CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        logger.info("=" * 30)
    
    @staticmethod
    def log_config_info(config_data: dict, config_name: str = "Config"):
        """Loggt Konfigurationsdaten strukturiert"""
        import json
        if _SUPPRESS_HEADER:
            return

        logger = logging.getLogger("ConfigInfo")
        logger.info(f"=== {config_name.upper()} ===")
        logger.info(json.dumps(config_data, indent=2, ensure_ascii=False))
        logger.info("=" * 30)
    
    @staticmethod
    def log_processing_session_start(audio_files: list):
        """Loggt Session-Start mit Datei-Informationen"""
        if _SUPPRESS_HEADER:
            return

        logger = logging.getLogger("SessionStart")

        logger.info("=== 🎵 PROCESSING SESSION START 🎵 ===")
        logger.info(f"Total files to process: {len(audio_files)}")

        for i, file_path in enumerate(audio_files, 1):
            from pathlib import Path
            file_info = Path(file_path)
            try:
                size_mb = file_info.stat().st_size / (1024 * 1024)
                logger.info(f"  {i:2d}. {file_info.name} ({size_mb:.1f}MB)")
            except:
                logger.info(f"  {i:2d}. {file_info.name} (size unknown)")

        logger.info("=" * 40)
    
    @staticmethod
    def log_file_processing_result(audio_file: str, tags: list, success: bool, error: str = None):
        """Loggt Ergebnis einer einzelnen Datei-Verarbeitung"""
        logger = logging.getLogger("FileResult")
        
        filename = Path(audio_file).name
        
        if success:
            logger.info(f"✅ SUCCESS: {filename}")
            logger.info(f"  Generated tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
            logger.info(f"  Total tags: {len(tags)}")
            logger.debug(f"  All tags: {tags}")
        else:
            logger.error(f"❌ FAILED: {filename}")
            if error:
                logger.error(f" ❌ Error: {error}")
    
    @staticmethod
    def log_session_summary(successful: int, failed: int, total_time: float):
        """Loggt Session-Zusammenfassung"""
        logger = logging.getLogger("SessionSummary")
        
        logger.info("=== 🎵 PROCESSING SESSION SUMMARY 🎵 ===")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"🕒 Total time: {total_time:.1f} seconds")
        logger.info(f"⏱️ Average time per file: {total_time/(successful+failed):.1f}s")
        logger.info("=" * 40)


def setup_session_logging(session_name: Optional[str] = None, verbose: bool = False, suppress_header: bool = False) -> str:
    """
    Convenience-Funktion für Session-Setup
    
    Returns:
        Pfad zur Log-Datei für spätere Referenz
    """
    # Log-Level basierend auf Verbose-Flag
    console_level = logging.DEBUG if verbose else logging.INFO
    file_level = logging.DEBUG  # Datei immer mit Debug
    
    # Wenn kein Session-Name vorgegeben ist, hier einmalig erzeugen und überall verwenden
    generated_session = session_name or f"tagging_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Logger Setup
    # Setze globale Suppress-Einstellung falls über Aufruf oder ENV gewünscht
    try:
        global _SUPPRESS_HEADER
        # Priorisiere übergebenes Argument, sonst ENV-Fallback
        if suppress_header:
            _SUPPRESS_HEADER = True
        else:
            env_flag = os.environ.get("LOG_SUPPRESS_HEADER", "").lower() in ("1", "true", "yes")
            _SUPPRESS_HEADER = env_flag or False
    except Exception:
        _SUPPRESS_HEADER = bool(suppress_header)

    DualLogger.setup_logging(
        console_level=console_level,
        file_level=file_level,
        session_name=generated_session
    )

    # System-Info loggen (optional, abhängig von _SUPPRESS_HEADER)
    DualLogger.log_system_info()
    
    # Return Log-File Path (identisch zum in setup_logging verwendeten Namen)
    log_file = Path("logs") / f"{generated_session}.log"
    return str(log_file)


# Utility Functions für einfache Verwendung
def get_session_logger(name: str) -> logging.Logger:
    """Gibt einen Logger für eine spezifische Komponente zurück"""
    return logging.getLogger(name)

def log_exception(logger: logging.Logger, operation: str, exception: Exception):
    """Loggt Exception mit Kontext"""
    logger.error(f"Exception in {operation}: {type(exception).__name__}: {exception}")
    logger.debug(f"Exception traceback:", exc_info=True)

def _safe_filename(text: str) -> str:
    """Erstellt einen dateisicheren, kurzen Dateinamen-Teil."""
    text = (text or "").strip().lower()
    # Erlaube Buchstaben, Zahlen, Bindestrich und Unterstrich
    text = re.sub(r"[^a-z0-9_-]+", "-", text)[:64]
    return text or "untitled"

def _get_raw_session_dir() -> Path:
    """Gibt den RAW-Ordner für die aktuelle Session zurück (wird falls nötig erzeugt)."""
    session = _CURRENT_SESSION_NAME or f"tagging_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_dir = _RAW_BASE_DIR / session
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def save_raw_response(category: str, response: str, prompt: Optional[str] = None, label: Optional[str] = None) -> Optional[str]:
    """
    Speichert die Rohantwort des Modells in einer separaten Datei unter logs/raw/<session>/.

    Args:
        category: Kategorie-Name (z. B. "genre", "mood")
        response: Rohantwort als String
        prompt: Optionaler Prompt-Text (wird in Datei mit abgelegt)
        label: Optionaler Zusatz (z. B. "attempt1"/"retry"/Dateiname)

    Returns:
        Pfad zur geschriebenen Datei oder None bei Fehler
    """
    try:
        if not isinstance(response, str):
            response = str(response)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cat = _safe_filename(category)
        lab = f"_{_safe_filename(label)}" if label else ""
        filename = f"{ts}_{cat}{lab}.txt"
        raw_dir = _get_raw_session_dir()
        path = raw_dir / filename

        with path.open('w', encoding='utf-8') as f:
            f.write(f"CATEGORY: {category}\n")
            f.write(f"TIMESTAMP: {ts}\n")
            f.write("=" * 80 + "\n\n")
            if prompt:
                f.write("PROMPT:\n")
                f.write(prompt)
                f.write("\n\n" + "-" * 80 + "\n\n")
            f.write("RAW RESPONSE:\n")
            f.write(response)
            f.write("\n")

        return str(path)
    except Exception as e:
        logging.getLogger("RawSaver").debug(f"Failed to save raw response: {e}")
        return None

def log_model_response(logger: logging.Logger, category: str, prompt: str, response: str, parsed: dict = None):
    """Loggt Model-Response detailliert für Debugging"""
    logger.debug(f"=== MODEL RESPONSE: {category} ===")
    logger.debug(f"Prompt length: {len(prompt)} chars")
    logger.debug(f"Prompt preview: {prompt[:200]}...")
    logger.debug(f"Raw response: {response}")
    logger.debug(f"Parsed result: {parsed}")
    logger.debug("=" * 50)
    # Zusätzlich: RAW-Response in separater Datei ablegen
    saved = save_raw_response(category=category, response=response, prompt=prompt)
    if saved:
        logger.debug(f"Raw response saved to: {saved}")


if __name__ == "__main__":
    # Test das Logging-System
    log_file = setup_session_logging("test_session", verbose=True)
    
    logger = get_session_logger("Test")
    logger.info("This is a test message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    DualLogger.log_config_info({"test": "value", "number": 42}, "Test Config")
    
    print(f"\nLog file created: {log_file}")
    print("Check the log file to see all messages!")
