#!/usr/bin/env python3
"""
Database backup script for the Mathematical Multimodal LLM System.

This script creates backups of the MongoDB database and handles rotation
of older backups.
"""
import os
import sys
import logging
import argparse
import subprocess
import datetime
import shutil
import glob
import tarfile
import json
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongodb_config import get_mongo_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('db_backup.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MongoDB backup script')
    
    parser.add_argument('--backup-dir', type=str, default='./backups',
                        help='Directory to store backups')
    parser.add_argument('--environment', type=str, default=None,
                        choices=['development', 'staging', 'production'],
                        help='Environment to back up')
    parser.add_argument('--max-backups', type=int, default=7,
                        help='Maximum number of backups to keep')
    parser.add_argument('--compress', action='store_true',
                        help='Compress the backup')
    parser.add_argument('--add-timestamp', action='store_true',
                        help='Add timestamp to backup directory name')
    
    return parser.parse_args()

def create_backup(
    mongo_config: Dict[str, Any],
    backup_dir: str,
    add_timestamp: bool = True,
    compress: bool = True
) -> Optional[str]:
    """
    Create a MongoDB backup.
    
    Args:
        mongo_config: MongoDB configuration
        backup_dir: Directory to store the backup
        add_timestamp: Add timestamp to backup directory name
        compress: Compress the backup
        
    Returns:
        Path to the backup directory, or None if failed
    """
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup path
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{mongo_config['database']}"
        
        if add_timestamp:
            backup_name = f"{backup_name}_{timestamp}"
        
        backup_path = os.path.join(backup_dir, backup_name)
        
        # Ensure the backup directory doesn't already exist
        if os.path.exists(backup_path):
            logger.warning(f"Backup directory already exists: {backup_path}")
            backup_path = f"{backup_path}_{timestamp}"
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Build mongodump command
        mongodump_cmd = [
            'mongodump',
            '--uri', mongo_config['uri'],
            '--db', mongo_config['database'],
            '--out', backup_path
        ]
        
        # Execute mongodump
        logger.info(f"Running mongodump: {' '.join(mongodump_cmd)}")
        result = subprocess.run(mongodump_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"mongodump failed: {result.stderr}")
            return None
        
        logger.info(f"Backup created at: {backup_path}")
        
        # Write backup metadata
        metadata = {
            "database": mongo_config['database'],
            "created_at": timestamp,
            "environment": os.environ.get("APP_ENV", "development"),
            "command": ' '.join(mongodump_cmd)
        }
        
        with open(os.path.join(backup_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Compress if requested
        if compress:
            # Create archive name
            archive_name = f"{backup_path}.tar.gz"
            
            logger.info(f"Compressing backup to: {archive_name}")
            
            # Create tar archive
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(backup_path, arcname=os.path.basename(backup_path))
            
            # Remove original directory
            shutil.rmtree(backup_path)
            
            logger.info(f"Backup compressed to: {archive_name}")
            return archive_name
        
        return backup_path
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None

def rotate_backups(backup_dir: str, database: str, max_backups: int) -> int:
    """
    Rotate old backups to keep only the specified number.
    
    Args:
        backup_dir: Backup directory
        database: Database name
        max_backups: Maximum number of backups to keep
        
    Returns:
        Number of backups removed
    """
    try:
        # Get list of backups
        backup_pattern = os.path.join(backup_dir, f"{database}*")
        backups = glob.glob(backup_pattern)
        
        # If we have fewer backups than the maximum, do nothing
        if len(backups) <= max_backups:
            logger.info(f"Found {len(backups)} backups, no rotation needed")
            return 0
        
        # Sort backups by modification time (oldest first)
        backups.sort(key=os.path.getmtime)
        
        # Calculate how many to remove
        to_remove = len(backups) - max_backups
        
        # Remove oldest backups
        for i in range(to_remove):
            backup = backups[i]
            logger.info(f"Removing old backup: {backup}")
            
            if os.path.isdir(backup):
                shutil.rmtree(backup)
            else:
                os.remove(backup)
        
        logger.info(f"Removed {to_remove} old backups")
        return to_remove
        
    except Exception as e:
        logger.error(f"Backup rotation failed: {e}")
        return 0

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get MongoDB configuration
    mongo_config = get_mongo_config(args.environment)
    
    # Create backup
    backup_path = create_backup(
        mongo_config=mongo_config,
        backup_dir=args.backup_dir,
        add_timestamp=args.add_timestamp,
        compress=args.compress
    )
    
    if not backup_path:
        logger.error("Backup failed")
        sys.exit(1)
    
    # Rotate old backups
    rotate_backups(
        backup_dir=args.backup_dir,
        database=mongo_config['database'],
        max_backups=args.max_backups
    )
    
    logger.info("Backup completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()
