"""Shared exception types for pdf2beamer."""


class Pdf2BeamerError(Exception):
    """Base class for project-specific errors."""


class OptionalDependencyNotInstalledError(Pdf2BeamerError):
    """Raised when an optional local backend dependency is unavailable."""


class LocalModelLoadError(Pdf2BeamerError):
    """Raised when a local model cannot be found or loaded."""


class LocalModelInferenceError(Pdf2BeamerError):
    """Raised when local model inference fails."""


class InvalidModelConfigurationError(Pdf2BeamerError):
    """Raised when model backend configuration is incomplete or unsupported."""
