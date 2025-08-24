"""
scripts/core/audio_processor.py
Intelligenter Audio-Prozessor für das modulare Audio-Tagging-System

Features:
- Optimiert für RTX 4070 Super (12GB VRAM) + 64GB RAM
- Intelligentes Caching mit MD5-Hash-Keys
- MP3 → 16kHz WAV Konvertierung mit Librosa
- 30-Sekunden Audio-Chunks für Qwen2-Audio
- Robuste Error-Behandlung und Memory-Management
- Batch-Processing Support für Multiple Files
"""

import os
import hashlib
import json
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import librosa
import soundfile as sf

# Setup Logging
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Konfiguration für Audio-Processing"""
    target_sr: int = 16000          # Qwen2-Audio erwartet 16kHz
    max_duration: float = 30.0      # Max 30 Sekunden für Model
    cache_dir: str = "data/cache"
    enable_compression: bool = True  # Gzip-Kompression für Cache
    normalize_audio: bool = True    # Audio-Normalisierung
    mono_conversion: bool = True    # Stereo → Mono
    cache_max_size_gb: float = 5.0  # Max Cache-Größe
    segment_strategy: str = "middle" # "start", "middle", "end", "best"
    num_workers: int = 4  # Default number of worker threads for batch processing
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'AudioConfig':
        """Lädt Konfiguration aus JSON-Datei"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Mappe relevante Felder
            audio_config = {}
            if 'sampling_rate' in config_data:
                audio_config['target_sr'] = config_data['sampling_rate']
            if 'audio_max_duration' in config_data:
                audio_config['max_duration'] = config_data['audio_max_duration']
            # Optional worker configuration
            if 'num_workers' in config_data:
                audio_config['num_workers'] = config_data['num_workers']
            
            return cls(**audio_config)
        except Exception as e:
            logger.warning(f"Could not load audio config from {config_path}: {e}")
            return cls()

@dataclass  
class ProcessedAudio:
    """Container für verarbeitete Audio-Daten"""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    source_path: str
    cache_path: Optional[str] = None
    processing_info: Optional[Dict] = None

class AudioProcessor:
    """
    Haupt Audio-Prozessor mit intelligentem Caching
    
    Optimiert für:
    - Qwen2-Audio-7B-Instruct (16kHz, Mono, max 30s)
    - RTX 4070 Super Performance
    - Minimal Memory Footprint
    """
    
    def __init__(self, config: Optional[AudioConfig] = None, cache_dir: Optional[str] = None):
        self.config = config or AudioConfig()
        
        # Cache Directory Setup
        if cache_dir:
            self.config.cache_dir = cache_dir
        
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache-Info-File für Metadaten
        self.cache_info_file = self.cache_dir / "cache_info.json"
        self.cache_info = self._load_cache_info()
        
        logger.info(f"AudioProcessor initialized: cache_dir={self.cache_dir}")
    
    def _load_cache_info(self) -> Dict[str, Any]:
        """Lädt Cache-Metadaten"""
        try:
            if self.cache_info_file.exists():
                with open(self.cache_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Cache info loading failed: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "cache_entries": {},
            "total_size_bytes": 0,
            "last_cleanup": None
        }
    
    def _save_cache_info(self):
        """Speichert Cache-Metadaten"""
        try:
            self.cache_info["last_updated"] = datetime.now().isoformat()
            with open(self.cache_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache info saving failed: {e}")
    
    def _generate_cache_key(self, file_path: str, processing_params: Dict) -> str:
        """
        Generiert eindeutigen Cache-Key basierend auf:
        - File-Pfad und Modification-Time
        - Processing-Parameter (SR, Duration, etc.)
        """
        try:
            file_stat = Path(file_path).stat()
            key_data = {
                "path": str(file_path),
                "size": file_stat.st_size,
                "mtime": file_stat.st_mtime,
                "params": processing_params
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Cache key generation failed: {e}")
            # Fallback: nur Filename + Parameter
            fallback_data = f"{Path(file_path).name}_{processing_params}"
            return hashlib.md5(fallback_data.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[ProcessedAudio]:
        """Lädt verarbeitetes Audio aus Cache"""
        try:
            cache_entry = self.cache_info["cache_entries"].get(cache_key)
            if not cache_entry:
                return None
            
            cache_path = self.cache_dir / cache_entry["filename"]
            if not cache_path.exists():
                # Cache-Entry existiert nicht mehr → bereinigen
                del self.cache_info["cache_entries"][cache_key]
                self._save_cache_info()
                return None
            
            # Audio laden
            if cache_entry.get("compressed", False):
                with gzip.open(cache_path, 'rb') as f:
                    audio_data = np.load(f)
            else:
                audio_data, sr = sf.read(cache_path)
                if sr != self.config.target_sr:
                    logger.warning(f"Cache SR mismatch: {sr} vs {self.config.target_sr}")
                    return None
            
            # Erfolgreich geladen → ProcessedAudio erstellen
            processed = ProcessedAudio(
                audio_data=audio_data,
                sample_rate=self.config.target_sr,
                duration=len(audio_data) / self.config.target_sr,
                source_path=cache_entry["source_path"],
                cache_path=str(cache_path),
                processing_info=cache_entry.get("processing_info", {})
            )
            
            logger.debug(f"Cache hit: {cache_key[:8]}... → {cache_path.name}")
            return processed
            
        except Exception as e:
            logger.warning(f"Cache loading failed for {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, processed: ProcessedAudio, source_path: str):
        """Speichert verarbeitetes Audio in Cache"""
        try:
            # Cleanup wenn Cache zu groß wird
            self._cleanup_cache_if_needed()
            
            # Cache-Filename generieren
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_name = Path(source_path).stem
            cache_filename = f"{source_name}_{cache_key[:8]}_{timestamp}"
            
            if self.config.enable_compression:
                # Speichere komprimiertes NumPy-Array im .npy.gz-Format
                cache_filename += ".npy.gz"
                cache_path = self.cache_dir / cache_filename
                with gzip.open(cache_path, 'wb') as f:
                    np.save(f, processed.audio_data)
            else:
                cache_filename += ".wav"
                cache_path = self.cache_dir / cache_filename
                sf.write(cache_path, processed.audio_data, self.config.target_sr)
            
            # Cache-Info updaten
            file_size = cache_path.stat().st_size
            cache_entry = {
                "filename": cache_filename,
                "source_path": source_path,
                "created": datetime.now().isoformat(),
                "size_bytes": file_size,
                "compressed": self.config.enable_compression,
                "processing_info": processed.processing_info or {}
            }
            
            self.cache_info["cache_entries"][cache_key] = cache_entry
            self.cache_info["total_size_bytes"] += file_size
            self._save_cache_info()
            
            logger.debug(f"Cached: {source_name} → {cache_filename} ({file_size/1024:.1f} KB)")
            
        except Exception as e:
            logger.error(f"Cache saving failed for {cache_key}: {e}")
    
    def _cleanup_cache_if_needed(self):
        """Bereinigt Cache wenn Größenlimit erreicht"""
        try:
            total_size_gb = self.cache_info["total_size_bytes"] / (1024**3)
            
            if total_size_gb <= self.config.cache_max_size_gb:
                return
            
            logger.info(f"Cache cleanup needed: {total_size_gb:.2f}GB > {self.config.cache_max_size_gb}GB")
            
            # Sortiere nach Erstellungsdatum (älteste zuerst)
            entries = list(self.cache_info["cache_entries"].items())
            entries.sort(key=lambda x: x[1]["created"])
            
            # Lösche älteste Einträge bis unter Limit
            freed_bytes = 0
            while total_size_gb > self.config.cache_max_size_gb * 0.8 and entries:  # 80% Ziel
                cache_key, entry = entries.pop(0)
                
                cache_path = self.cache_dir / entry["filename"]
                if cache_path.exists():
                    file_size = entry["size_bytes"]
                    cache_path.unlink()
                    freed_bytes += file_size
                    logger.debug(f"Cleaned: {entry['filename']}")
                
                del self.cache_info["cache_entries"][cache_key]
                self.cache_info["total_size_bytes"] -= entry["size_bytes"]
                total_size_gb = self.cache_info["total_size_bytes"] / (1024**3)
            
            self.cache_info["last_cleanup"] = datetime.now().isoformat()
            self._save_cache_info()
            
            logger.info(f"Cache cleanup completed: freed {freed_bytes/(1024**2):.1f}MB")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _select_audio_segment(self, audio_data: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Intelligente Auswahl des besten Audio-Segments
        
        Strategien:
        - 'middle': Mittlerer Teil (Standard)
        - 'start': Anfang (Intro-Analyse)
        - 'end': Ende (Outro-Analyse)  
        - 'best': Dynamischer Teil mit höchster RMS-Energy
        """
        duration = len(audio_data) / sr
        max_duration = self.config.max_duration
        
        processing_info = {
            "original_duration": duration,
            "segment_strategy": self.config.segment_strategy,
            "target_duration": max_duration
        }
        
        if duration <= max_duration:
            # Audio ist bereits kurz genug
            processing_info["segment_used"] = "full"
            processing_info["segment_start"] = 0.0
            return audio_data, processing_info
        
        max_samples = int(max_duration * sr)
        
        if self.config.segment_strategy == "start":
            segment = audio_data[:max_samples]
            start_time = 0.0
            
        elif self.config.segment_strategy == "end":
            segment = audio_data[-max_samples:]
            start_time = duration - max_duration
            
        elif self.config.segment_strategy == "best":
            # Finde Segment mit höchster RMS-Energy
            segment, start_time = self._find_best_segment(audio_data, sr, max_duration)
            
        else:  # "middle" (default)
            start_sample = (len(audio_data) - max_samples) // 2
            segment = audio_data[start_sample:start_sample + max_samples]
            start_time = start_sample / sr
        
        processing_info["segment_used"] = f"{max_duration}s_{self.config.segment_strategy}"
        processing_info["segment_start"] = start_time
        
        return segment, processing_info

    def _select_audio_segment_with_strategy(self, audio_data: np.ndarray, sr: int, strategy: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wie _select_audio_segment, aber mit expliziter Strategiewahl pro Aufruf."""
        original_strategy = self.config.segment_strategy
        try:
            self.config.segment_strategy = strategy
            return self._select_audio_segment(audio_data, sr)
        finally:
            self.config.segment_strategy = original_strategy
    
    def _find_best_segment(self, audio_data: np.ndarray, sr: int, duration: float) -> Tuple[np.ndarray, float]:
        """Findet das Segment mit der höchsten durchschnittlichen Energy"""
        try:
            max_samples = int(duration * sr)
            hop_size = sr // 4  # 0.25s Schritte
            
            best_rms = 0
            best_start = 0
            
            for start_sample in range(0, len(audio_data) - max_samples + 1, hop_size):
                segment = audio_data[start_sample:start_sample + max_samples]
                rms = np.sqrt(np.mean(segment**2))
                
                if rms > best_rms:
                    best_rms = rms
                    best_start = start_sample
            
            segment = audio_data[best_start:best_start + max_samples]
            start_time = best_start / sr
            
            logger.debug(f"Best segment: {start_time:.1f}s-{start_time + duration:.1f}s (RMS: {best_rms:.4f})")
            return segment, start_time
            
        except Exception as e:
            logger.warning(f"Best segment detection failed: {e}, using middle")
            # Fallback zu middle
            start_sample = (len(audio_data) - max_samples) // 2
            return audio_data[start_sample:start_sample + max_samples], start_sample / sr

    def _normalize_segment(self, segment: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Defensive Normalisierung: Clip-Schutz, begrenztes Gain, keine Anhebung sehr leiser Segmente."""
        try:
            if segment.size == 0:
                info["normalized"] = False
                info["gain_db"] = 0.0
                info["peak_before"] = 0.0
                info["rms_before"] = 0.0
                return segment

            peak = float(np.max(np.abs(segment)))
            rms = float(np.sqrt(np.mean(segment ** 2)))
            info["peak_before"] = peak
            info["rms_before"] = rms

            # 1) Nur absenken, wenn Peak > 1.0 (Clipping-Schutz)
            if peak > 1.0:
                g = 1.0 / peak
                segment = segment * g
                info["normalized"] = True
                info["gain_db"] = float(20.0 * np.log10(max(g, 1e-12)))
                return segment

            # 2) Sehr leise Segmente nicht hochziehen (Rauschen vermeiden)
            # Schwellenwerte: Peak < 0.2 (~ -14 dBFS) oder RMS < 0.02 (~ -34 dBFS @ float32)
            if peak < 0.2 or rms < 0.02:
                info["normalized"] = False
                info["gain_db"] = 0.0
                return segment

            # 3) Leichte Normalisierung mit Gain-Limit (max +6 dB), Zielpeak ~ -1 dBFS
            target_peak = 0.89
            current_peak = max(peak, 1e-9)
            g = min(target_peak / current_peak, 2.0)  # 2.0 == +6 dB

            if abs(g - 1.0) > 1e-3:
                segment = segment * g
                info["normalized"] = True
                info["gain_db"] = float(20.0 * np.log10(max(g, 1e-12)))
            else:
                info["normalized"] = False
                info["gain_db"] = 0.0

            return segment
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            info["normalized"] = False
            info["gain_db"] = 0.0
            return segment
    
    def process_audio_file(self, file_path: str, force_reprocess: bool = False) -> Optional[ProcessedAudio]:
        """
        Hauptmethode: Verarbeitet Audio-Datei mit intelligentem Caching
        
        Args:
            file_path: Pfad zur Audio-Datei
            force_reprocess: Cache ignorieren und neu verarbeiten
            
        Returns:
            ProcessedAudio-Objekt oder None bei Fehlern
        """
        try:
            file_path = str(Path(file_path).resolve())  # Normalisierter Pfad
            
            if not Path(file_path).exists():
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            # Processing-Parameter für Cache-Key
            processing_params = {
                "target_sr": self.config.target_sr,
                "max_duration": self.config.max_duration,
                "mono": self.config.mono_conversion,
                "normalize": self.config.normalize_audio,
                "segment_strategy": self.config.segment_strategy,
                "norm_version": 2,
                "resampler": "soxr_hq"
            }
            
            cache_key = self._generate_cache_key(file_path, processing_params)
            
            # Cache-Lookup (wenn nicht force_reprocess)
            if not force_reprocess:
                cached = self._get_cached_audio(cache_key)
                if cached:
                    return cached
            
            logger.info(f"Processing audio: {Path(file_path).name}")
            
            # Audio laden (hochwertiges Resampling mit Fallback)
            try:
                try:
                    audio_data, orig_sr = librosa.load(
                        file_path,
                        sr=self.config.target_sr if self.config.target_sr > 0 else None,
                        mono=self.config.mono_conversion,
                        res_type="soxr_hq"
                    )
                except Exception as e_soxr:
                    logger.debug(f"librosa.load soxr_hq failed ({e_soxr}), falling back to kaiser_best")
                    audio_data, orig_sr = librosa.load(
                        file_path,
                        sr=self.config.target_sr if self.config.target_sr > 0 else None,
                        mono=self.config.mono_conversion,
                        res_type="kaiser_best"
                    )
            except Exception as e:
                logger.error(f"Audio loading failed for {file_path}: {e}")
                return None
            
            if len(audio_data) == 0:
                logger.error(f"Empty audio file: {file_path}")
                return None
            
            # Segment-Auswahl falls zu lang
            segment_data, processing_info = self._select_audio_segment(audio_data, self.config.target_sr)
            
            # Normalisierung
            if self.config.normalize_audio:
                segment_data = self._normalize_segment(segment_data, processing_info)
            
            # ProcessedAudio erstellen
            processed = ProcessedAudio(
                audio_data=segment_data.astype(np.float32),
                sample_rate=self.config.target_sr,
                duration=len(segment_data) / self.config.target_sr,
                source_path=file_path,
                processing_info=processing_info
            )
            
            # In Cache speichern
            self._save_to_cache(cache_key, processed, file_path)
            
            logger.info(f"Audio processed: {processed.duration:.1f}s @ {processed.sample_rate}Hz")
            return processed
            
        except Exception as e:
            logger.error(f"Audio processing failed for {file_path}: {e}")
            return None

    def process_audio_segments(self, file_path: str, segments: List[str], force_reprocess: bool = False) -> List[ProcessedAudio]:
        """
        Verarbeitet mehrere 30s‑Segmente aus einer Datei und cached jede Variante separat.

        Unterstützte Segmentnamen: "start", "middle", "end", "best", "full".
        "full" wird nur verwendet, wenn die Datei kürzer als max_duration ist.
        """
        results: List[ProcessedAudio] = []
        try:
            file_path = str(Path(file_path).resolve())
            if not Path(file_path).exists():
                logger.error(f"Audio file not found: {file_path}")
                return results

            # Einmal laden/konvertieren (hochwertiges Resampling mit Fallback)
            try:
                try:
                    audio_data, _ = librosa.load(
                        file_path,
                        sr=self.config.target_sr if self.config.target_sr > 0 else None,
                        mono=self.config.mono_conversion,
                        res_type="soxr_hq"
                    )
                except Exception as e_soxr:
                    logger.debug(f"librosa.load soxr_hq failed ({e_soxr}), falling back to kaiser_best")
                    audio_data, _ = librosa.load(
                        file_path,
                        sr=self.config.target_sr if self.config.target_sr > 0 else None,
                        mono=self.config.mono_conversion,
                        res_type="kaiser_best"
                    )
            except Exception as e:
                logger.error(f"Audio loading failed for {file_path}: {e}")
                return results

            if len(audio_data) == 0:
                logger.error(f"Empty audio file: {file_path}")
                return results

            duration = len(audio_data) / self.config.target_sr

            for seg in segments:
                seg = (seg or "").lower().strip()
                # Volle Länge nur, wenn <= max_duration
                if seg == "full" and duration > self.config.max_duration:
                    continue

                # Cache-Key pro Strategie
                processing_params = {
                    "target_sr": self.config.target_sr,
                    "max_duration": self.config.max_duration,
                    "mono": self.config.mono_conversion,
                    "normalize": self.config.normalize_audio,
                    "segment_strategy": seg or self.config.segment_strategy,
                    "norm_version": 2,
                    "resampler": "soxr_hq",
                }
                cache_key = self._generate_cache_key(file_path, processing_params)

                if not force_reprocess:
                    cached = self._get_cached_audio(cache_key)
                    if cached:
                        results.append(cached)
                        continue

                # Segment wählen
                if seg in ("start", "middle", "end", "best"):
                    segment_data, processing_info = self._select_audio_segment_with_strategy(audio_data, self.config.target_sr, seg)
                elif seg == "full" and duration <= self.config.max_duration:
                    segment_data = audio_data
                    processing_info = {
                        "original_duration": duration,
                        "segment_strategy": "full",
                        "segment_used": "full",
                        "segment_start": 0.0,
                        "target_duration": duration,
                    }
                else:
                    # Fallback: nutze aktuelle Default-Strategie
                    segment_data, processing_info = self._select_audio_segment(audio_data, self.config.target_sr)

                # Normalisierung
                if self.config.normalize_audio:
                    segment_data = self._normalize_segment(segment_data, processing_info)

                processed = ProcessedAudio(
                    audio_data=segment_data.astype(np.float32),
                    sample_rate=self.config.target_sr,
                    duration=len(segment_data) / self.config.target_sr,
                    source_path=file_path,
                    processing_info=processing_info
                )

                self._save_to_cache(cache_key, processed, file_path)
                # Erneut aus Cache holen, damit cache_path gesetzt ist
                cached = self._get_cached_audio(cache_key)
                if cached:
                    results.append(cached)
                else:
                    results.append(processed)

            return results
        except Exception as e:
            logger.error(f"Audio multi-segment processing failed for {file_path}: {e}")
            return results
    
    def process_batch(self, file_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, Optional[ProcessedAudio]]:
        """
        Batch-Processing für mehrere Audio-Dateien
        Optimiert für Parallel-I/O. Wenn keine Threadanzahl angegeben ist,
        wird die Konfiguration aus dem AudioConfig verwendet.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Bestimme Worker-Anzahl: Priorisiere explizite Vorgabe, dann Config, sonst 4
        workers = max_workers or getattr(self.config, "num_workers", 4)
        logger.info(f"Batch processing {len(file_paths)} files with {workers} worker(s)")

        results: Dict[str, Optional[ProcessedAudio]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit alle Jobs
            future_to_path = {
                executor.submit(self.process_audio_file, path): path
                for path in file_paths
            }
            # Sammle Ergebnisse
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                except Exception as e:
                    logger.error(f"Batch processing failed for {path}: {e}")
                    results[path] = None

        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"Batch processing completed: {successful}/{len(file_paths)} successful")
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        return {
            "total_entries": len(self.cache_info["cache_entries"]),
            "total_size_mb": self.cache_info["total_size_bytes"] / (1024**2),
            "cache_dir": str(self.cache_dir),
            "last_cleanup": self.cache_info.get("last_cleanup"),
            "config": self.config.__dict__
        }
    
    def clear_cache(self):
        """Bereinigt kompletten Cache"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.cache_info = {
                "created": datetime.now().isoformat(),
                "cache_entries": {},
                "total_size_bytes": 0,
                "last_cleanup": datetime.now().isoformat()
            }
            self._save_cache_info()
            
            logger.info("Cache cleared completely")
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")


# Convenience Functions
def create_audio_processor(config_path: Optional[str] = None, **kwargs) -> AudioProcessor:
    """Factory-Funktion für AudioProcessor"""
    if config_path and Path(config_path).exists():
        config = AudioConfig.from_config_file(config_path)
    else:
        config = AudioConfig(**kwargs)
    
    return AudioProcessor(config)


def quick_process_audio(file_path: str, target_sr: int = 16000, max_duration: float = 30.0) -> Optional[ProcessedAudio]:
    """Schnelle Audio-Verarbeitung ohne Cache für Single-Use"""
    processor = AudioProcessor(AudioConfig(
        target_sr=target_sr,
        max_duration=max_duration,
        cache_dir=tempfile.mkdtemp()  # Temporärer Cache
    ))
    
    result = processor.process_audio_file(file_path)
    
    # Cleanup temporärer Cache
    try:
        shutil.rmtree(processor.cache_dir)
    except:
        pass
    
    return result


if __name__ == "__main__":
    # Test-Code für Development
    logging.basicConfig(level=logging.DEBUG)
    
    # Test mit Cache-System
    processor = AudioProcessor()
    print(f"Cache stats: {processor.get_cache_stats()}")
    
    # Hier können Sie Test-Audio-Dateien verwenden
    # result = processor.process_audio_file("test.mp3")
    # if result:
    #     print(f"Processed: {result.duration:.1f}s, shape: {result.audio_data.shape}")
