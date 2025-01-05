import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Change to DEBUG for detailed logging
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
