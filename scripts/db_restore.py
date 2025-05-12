#!/usr/bin/env python3
"""
Database restore script for the Mathematical Multimodal LLM System.

This script restores MongoDB backups created by the db_backup.py script.
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
import tempfile
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
        logging.FileHandler('db_restore.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MongoDB restore script')
    
    parser.add_argument('--backup-path', type=str, required=True,
                        help='Path to the backup file or directory')
    parser.add_argument('--environment', type=str, default=None,
                        choices=['development', 'staging', 'production'],
                        help='Environment to restore to')
    parser.add_argument('--drop', action='store_true',
                        help='Drop collections before restoring')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually restoring')
    
    return parser.parse_args()

def restore_backup(
    mongo_config: Dict[str, Any],
    backup_path: str,
    drop: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Restore a MongoDB backup.
    
    Args:
        mongo_config: MongoDB configuration
        backup_path: Path to the backup file or directory
        drop: Drop collections before restoring
        dry_run: Show what would be done without actually restoring
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if the backup exists
        if not os.path.exists(backup_path):
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        # If backup is a compressed file, extract it
        temp_dir = None
        if os.path.isfile(backup_path) and backup_path.endswith('.tar.gz'):
            logger.info(f"Extracting backup: {backup_path}")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Extract archive
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
            
            # Find extracted directory
            subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if not subdirs:
                logger.error("No directories found in backup archive")
                if temp_dir:
                    shutil.rmtree(temp_dir)
                return False
            
            # Use the first directory as backup path
            backup_path = os.path.join(temp_dir, subdirs[0])
            logger.info(f"Using extracted backup: {backup_path}")
        
        # Check if it's a mongodump directory
        db_dir = os.path.join(backup_path, mongo_config["database"])
        if not os.path.isdir(db_dir):
            logger.error(f"Backup directory structure is not as expected. Could not find: {db_dir}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return False
        
        # Build mongorestore command
        mongorestore_cmd = [
            'mongorestore',
            '--uri', mongo_config['uri'],
            '--db', mongo_config['database'],
            '--dir', db_dir
        ]
        
        if drop:
            mongorestore_cmd.append('--drop')
        
        # Log what would be done in dry run mode
        if dry_run:
            logger.info(f"DRY RUN: Would execute: {' '.join(mongorestore_cmd)}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return True
        
        # Execute mongorestore
        logger.info(f"Running mongorestore: {' '.join(mongorestore_cmd)}")
        result = subprocess.run(mongorestore_cmd, capture_output=True, text=True)
        
        # Clean up temporary directory
        if temp_dir:
            shutil.rmtree(temp_dir)
        
        if result.returncode != 0:
            logger.error(f"mongorestore failed: {result.stderr}")
            return False
        
        logger.info(f"Restore completed from: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get MongoDB configuration
    mongo_config = get_mongo_config(args.environment)
    
    # Confirm restore in production
    if args.environment == "production" and not args.dry_run:
        confirm = input(f"Are you sure you want to restore to the PRODUCTION database ({mongo_config['database']})? [y/N]: ")
        if confirm.lower() != "y":
            logger.info("Restore cancelled")
            sys.exit(0)
    
    # Restore backup
    success = restore_backup(
        mongo_config=mongo_config,
        backup_path=args.backup_path,
        drop=args.drop,
        dry_run=args.dry_run
    )
    
    if not success:
        logger.error("Restore failed")
        sys.exit(1)
    
    logger.info("Restore completed successfully" if not args.dry_run else "Dry run completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()
