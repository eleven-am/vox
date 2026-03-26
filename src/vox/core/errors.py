class VoxError(Exception):
    """Base exception for all Vox errors."""

class ModelNotFoundError(VoxError):
    """Raised when a model is not found locally or in the registry."""
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"model '{model}' not found")

class AdapterNotFoundError(VoxError):
    """Raised when no adapter is installed for the requested model architecture."""
    def __init__(self, adapter: str):
        self.adapter = adapter
        super().__init__(f"adapter '{adapter}' not installed. Install it with: pip install vox-{adapter}")

class ModelLoadError(VoxError):
    """Raised when a model fails to load."""

class TranscriptionError(VoxError):
    """Raised when transcription fails."""

class SynthesisError(VoxError):
    """Raised when speech synthesis fails."""

class OOMError(ModelLoadError):
    """Raised when a model fails to load due to out-of-memory."""

class PullError(VoxError):
    """Raised when model download fails."""
