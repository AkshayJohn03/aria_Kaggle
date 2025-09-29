from loguru import logger
import os

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Log format
logger.add("logs/run_{time}.log", rotation="1 MB", retention="10 days", compression="zip")

# Example usage
logger.info("Logger initialized")
