#!/usr/bin/env python3
"""
Set up the database schema.

This script initializes the MongoDB schema for the
Mathematical Multimodal LLM System.
"""
import os
import sys
import logging
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongodb_config import get_mongo_config
from database.schema.schema_manager import SchemaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_database.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up database schema')
    
    parser.add_argument('--environment', type=str, default=None,
                        choices=['development', 'staging', 'production'],
                        help='Environment for configuration')
    parser.add_argument('--connection-string', type=str, default=None,
                        help='MongoDB connection string (overrides configuration)')
    parser.add_argument('--database', type=str, default=None,
                        help='Database name (overrides configuration)')
    parser.add_argument('--force', action='store_true',
                        help='Force schema initialization even if already initialized')
    parser.add_argument('--check-integrity', action='store_true',
                        help='Check data integrity after initialization')
    parser.add_argument('--maintenance', action='store_true',
                        help='Perform database maintenance after initialization')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get MongoDB configuration
    mongo_config = get_mongo_config(args.environment)
    
    # Override with command line arguments
    connection_string = args.connection_string or mongo_config['uri']
    database = args.database or mongo_config['database']
    
    # Create schema manager
    schema_manager = SchemaManager(connection_string, database)
    
    # Initialize schema
    logger.info(f"Initializing schema for database: {database}")
    success = schema_manager.initialize_schema(force=args.force)
    
    if not success:
        logger.error("Failed to initialize schema")
        sys.exit(1)
    
    # Get current schema version
    version = schema_manager.get_current_schema_version()
    logger.info(f"Schema initialized successfully (version {version})")
    
    # Check data integrity if requested
    if args.check_integrity:
        logger.info("Checking data integrity...")
        results = schema_manager.check_data_integrity()
        
        # Log results
        logger.info(f"Schema version: {results['schema_version']}")
        logger.info(f"Database collections: {len(results['collections'])}")
        
        if results['issues']:
            logger.warning(f"Found {len(results['issues'])} issues")
            for issue in results['issues']:
                logger.warning(f"Issue: {issue}")
        else:
            logger.info("No issues found")
    
    # Perform maintenance if requested
    if args.maintenance:
        logger.info("Performing database maintenance...")
        results = schema_manager.perform_maintenance()
        
        # Log results
        logger.info(f"Maintenance actions: {len(results['actions'])}")
        
        if results['issues']:
            logger.warning(f"Encountered {len(results['issues'])} issues during maintenance")
            for issue in results['issues']:
                logger.warning(f"Issue: {issue}")
        else:
            logger.info("Maintenance completed successfully")
    
    logger.info("Database setup completed")
    sys.exit(0)

if __name__ == "__main__":
    main()
