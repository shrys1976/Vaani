class VaaniError(Exception):
    """Base exception for the application."""


class IntentValidationError(VaaniError):
    """Raised when the LLM output cannot be validated as a supported intent."""


class LLMServiceError(VaaniError):
    """Raised when the LLM provider or parsing pipeline fails."""


class STTServiceError(VaaniError):
    """Raised when speech-to-text processing fails."""


class ToolExecutionError(VaaniError):
    """Raised when a local tool cannot safely complete the requested action."""
