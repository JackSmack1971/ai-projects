import logging
import structlog

def configure_logging() -> None:
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Use standard logging directly for now to bypass structlog issues
logger = logging.getLogger(__name__)