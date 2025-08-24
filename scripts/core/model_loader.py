"""
scripts/core/model_loader.py

Optimised model loader for the Qwen2‑Audio‑7B‑Instruct model using 4‑bit quantisation.
This module integrates the model into a modular audio tagging architecture and
handles all of the boilerplate around loading, quantising and chatting with
the model.  It supports automatic 4‑bit NF4 quantisation with a fallback
to FP16 if bitsandbytes is unavailable, environment variable controls for
runtime configuration, ChatML template support for structured prompts and an
optimised audio preprocessing pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
import os
import json
import logging

import numpy as np
import librosa
import torch

# Hugging Face imports with graceful fallback
try:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    BitsAndBytesConfig = None
    HAS_BITSANDBYTES = False

# Logging Setup
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for loading the Qwen2‑Audio model.

    These fields control which model to load, how many tokens to generate
    and whether to use 4‑bit quantisation or force FP16.  They also expose
    device mapping and dtype options.  Unknown keys in the JSON
    configuration file are ignored.
    """
    model_path: str = "Qwen/Qwen2-Audio-7B-Instruct"
    max_new_tokens: int = 80
    temperature: float = 0.2
    top_p: float = 0.85
    use_4bit: bool = True
    force_fp16: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from a JSON file.

        Any keys in the JSON that correspond to attributes on
        :class:`ModelConfig` are used to override the defaults.  On error a
        default configuration is returned.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls(**{k: v for k, v in config_data.items() if hasattr(cls, k)})
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return cls()


class Qwen2AudioWrapper:
    """
    Modern wrapper around the Qwen2‑Audio model with an optimised chat API.

    This class exposes a ``chat`` method compatible with existing code while
    adding support for structured tag generation workflows.  It manages
    preprocessing of audio inputs, ChatML template construction and model
    invocation.  Model loading, quantisation and device selection are
    configured via :class:`ModelConfig`.
    """
    
    def __init__(self, model: Qwen2AudioForConditionalGeneration, processor: AutoProcessor, config: ModelConfig):
        self.model = model.eval()
        self.processor = processor
        self.config = config
        self.sr = processor.feature_extractor.sampling_rate  # Standard: 16000 Hz
        
        logger.info(f"Qwen2-Audio loaded: sampling_rate={self.sr}Hz, device={self.device}")

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is allocated.

        If obtaining ``self.model.device`` fails, fall back to the device of
        the first model parameter. This ensures that tensor operations
        continue to target a valid device.
        """
        try:
            return self.model.device
        except Exception:
            # Fallback: return the device of the first model parameter
            return next(self.model.parameters()).device

    def generate(self, *args, **kwargs):
        """Direct access to ``model.generate`` for backward compatibility."""
        return self.model.generate(*args, **kwargs)

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess an audio file for inference.

        The audio is resampled to the model's target sample rate, converted
        to mono and cast to ``float32``.  A peak normalisation step is
        performed only if the maximum absolute value exceeds 1.0 to avoid
        clipping.  Any errors during loading are propagated.
        """
        try:
            # Load audio with the correct sample rate
            audio_data, _ = librosa.load(audio_path, sr=self.sr, mono=True)

            # Cast to float32 for better precision
            audio_data = audio_data.astype(np.float32)

            # Perform peak normalisation only if the maximum value exceeds 1.0 (prevents clipping)
            max_val = float(np.max(np.abs(audio_data))) if audio_data.size > 0 else 1.0
            if max_val > 1.0:
                audio_data = audio_data / max_val

            return audio_data

        except Exception as e:
            logger.error(f"Audio preprocessing failed for {audio_path}: {e}")
            raise

    @torch.inference_mode()
    def chat(
        self,
        tokenizer: Any = None,  # Placeholder for backward compatibility
        prompt: str = "",
        audio_files: Optional[List[str]] = None,
        segs: Any = None,  # Unused legacy parameter for compatibility
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Perform a chat inference with optional audio input.

        This method constructs a ChatML conversation containing the provided
        prompt and any supplied audio files, applies the model's chat template
        and then generates a textual response.  The number of new tokens and
        sampling parameters can be overridden per call.

        Args:
            prompt: the text prompt describing the analysis task.
            audio_files: an optional list of audio file paths or a single path
                to include in the conversation.
            max_new_tokens: optional override for the maximum number of tokens
                to generate.
            temperature: optional override for the sampling temperature.
            top_p: optional override for nucleus sampling.

        Returns:
            The generated text response.
        """
        
        # Combine call‑time overrides with configuration defaults
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        top_p_val = top_p if top_p is not None else self.config.top_p
        
        # Preprocess any provided audio files
        audio_data: List[np.ndarray] = []
        if audio_files:
            files = audio_files if isinstance(audio_files, list) else [audio_files]
            for audio_path in files:
                try:
                    processed_audio = self.preprocess_audio(audio_path)
                    audio_data.append(processed_audio)
                    logger.debug(f"Processed audio: {audio_path} -> {processed_audio.shape}")
                except Exception as e:
                    logger.warning(f"Skipping audio file {audio_path}: {e}")
                    continue
        
        # Provide a sensible fallback prompt if none was supplied
        if not prompt:
            prompt = "Analyze this audio and generate comprehensive musical tags as JSON with genre, mood, instruments, and technical characteristics."
        
        # Build the ChatML conversation structure
        # Build audio content entries: one per audio clip so tokens match
        audio_contents = []
        if audio_data:
            audio_contents = [{"type": "audio", "audio_url": "file://local"} for _ in range(len(audio_data))]

        conversation = [{
            "role": "user",
            "content": (
                audio_contents + [{"type": "text", "text": prompt}]
            ),
        }]
        
        # Apply the chat template
        try:
            formatted_text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
        except Exception as e:
            logger.error(f"Chat template failed: {e}")
            # Fall back to using the raw prompt when template application fails
            formatted_text = prompt
        
        # Prepare model inputs via the processor
        try:
            # Einige Processor-Versionen erwarten 'audio' statt 'audios'. Wir entscheiden dynamisch.
            processor_kwargs: Dict[str, Any] = {
                "text": formatted_text,
                "return_tensors": "pt",
                "padding": True,
            }
            if audio_data:
                # Bevorzugt 'audio' (einzelnes oder Liste), falle auf 'audios' zurück, falls unterstützt
                try:
                    inputs = self.processor(
                        audio=audio_data if len(audio_data) > 1 else audio_data[0],
                        sampling_rate=self.sr,
                        **processor_kwargs,
                    )
                except TypeError:
                    # Fallback auf 'audios' falls 'audio' unbekannt
                    inputs = self.processor(
                        audios=audio_data,
                        sampling_rate=self.sr,
                        **processor_kwargs,
                    )
            else:
                inputs = self.processor(**processor_kwargs)
        except Exception as e:
            logger.error(f"Processor failed: {e}")
            raise
        
        # Move the tensors onto the correct device
        self._move_tensors_to_device(inputs)
        
        # Generation Parameters
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "pad_token_id": getattr(self.processor.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.processor.tokenizer, "eos_token_id", None),
        }
        
        # Enable sampling only when the temperature is greater than zero
        if temp and temp > 0.0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temp),
                "top_p": float(top_p_val),
            })
        else:
            gen_kwargs["do_sample"] = False
        
        try:
            # Model generation
            # Use mixed precision only when running on a CUDA device for efficiency
            # Verwende neue API-Empfehlung für Autocast
            use_cuda = (self.device.type == "cuda")
            ctx = (torch.amp.autocast("cuda") if use_cuda else torch.cpu.amp.autocast()) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_cuda)
            with ctx:
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode only the newly generated tokens (exclude the input tokens)
            new_tokens = output_ids[:, inputs.input_ids.size(1):]
            
            # Decode tokens into text
            generated_text = self.processor.batch_decode(
                new_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _move_tensors_to_device(self, inputs: Dict[str, Any]) -> None:
        """Helper method to move all tensors onto the correct device."""
        target_device = self.device
        
        if hasattr(inputs, "items"):
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(target_device)
        elif hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            inputs.input_ids = inputs.input_ids.to(target_device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return debugging information about the loaded model."""
        return {
            "model_class": self.model.__class__.__name__,
            "device": str(self.device),
            "sampling_rate": self.sr,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else 'unknown',
            "config": self.config.__dict__,
        }


def load_qwen2audio_model(
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    **override_params
) -> Tuple[Qwen2AudioWrapper, Any]:
    """
    Main function for loading the Qwen2‑Audio model.

    Args:
        config_path: Optional path to a model_config.json file.
        model_path: Optional override for the model path.
        **override_params: Direct parameter overrides applied to the configuration.

    Returns:
        A tuple of (wrapper, tokenizer_placeholder).
    """
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = ModelConfig.from_config_file(config_path)
    else:
        config = ModelConfig()
    
    # Apply parameter overrides
    if model_path:
        config.model_path = model_path
    for key, value in override_params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info(f"Loading Qwen2-Audio: {config.model_path}")
    
    # Quantization Setup
    dtype = torch.float16 if config.torch_dtype == "float16" else torch.float32
    load_kwargs = {
        "device_map": config.device_map,
        "torch_dtype": dtype
    }
    
    # 4-bit quantisation if possible and desired
    use_4bit = (
        HAS_BITSANDBYTES and 
        config.use_4bit and 
        not config.force_fp16 and
        not _env_force_fp16()
    )
    
    if use_4bit:
        logger.info("Using 4-bit quantization (NF4, double-quant)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        logger.info(f"Using {dtype} precision (4-bit not available/disabled)")
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(config.model_path)
        
        # Load model
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.model_path, 
            **load_kwargs
        )
        
        # Create wrapper
        wrapper = Qwen2AudioWrapper(model, processor, config)
        
        # Tokenizer placeholder for compatibility
        tokenizer = getattr(processor, "tokenizer", object())
        
        logger.info("Model loading successful")
        logger.debug(f"Model info: {wrapper.get_model_info()}")
        
        return wrapper, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def _env_force_fp16() -> bool:
    """Check the environment variable that forces FP16 precision."""
    return os.environ.get("QWEN2_FORCE_FP16", "0").lower() in ("1", "true", "yes")


def get_model_requirements() -> Dict[str, Any]:
    """
    Return hardware requirements and recommendations.

    This information can be used for UI display and diagnostics.
    """
    return {
        # Minimum VRAM required when using 4‑bit quantization
        "min_vram_gb": 6,
        # Recommended VRAM for optimal performance
        "recommended_vram_gb": 16,
        "min_ram_gb": 16,
        "recommended_ram_gb": 32,
        "requires_cuda": True,
        # CPU inference is technically possible but not practically usable
        "supports_cpu": False,
        "quantization_available": HAS_BITSANDBYTES,
        "estimated_load_time_seconds": 30,
    }


# Convenience functions for easy integration
def quick_load_model(model_path: str = "Qwen/Qwen2-Audio-7B-Instruct") -> Qwen2AudioWrapper:
    """Quickly load the model with default parameters."""
    wrapper, _ = load_qwen2audio_model(model_path=model_path)
    return wrapper


if __name__ == "__main__":
    # Test code for development
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        wrapper = quick_load_model()
        print("Model loaded successfully!")
        print(f"Model info: {wrapper.get_model_info()}")
        
        # Test ohne Audio
        result = wrapper.chat(prompt="Hello, can you hear me?")
        print(f"Test response: {result}")
        
    except Exception as e:
        print(f"Test failed: {e}")
