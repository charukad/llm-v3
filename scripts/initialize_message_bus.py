#!/usr/bin/env python3
"""
Initialize the message bus configuration.

This script sets up the RabbitMQ exchanges, queues, and bindings
for the Mathematical Multimodal LLM System.
"""
import os
import sys
import logging
import argparse
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.rabbitmq_config import (
    get_rabbitmq_config,
    get_exchange_config,
    get_queue_config,
    get_routing_key,
    EXCHANGES,
    QUEUES
)
import pika
from pika.exceptions import AMQPConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('initialize_message_bus.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Initialize message bus configuration')
    
    parser.add_argument('--environment', type=str, default=None,
                        choices=['development', 'staging', 'production'],
                        help='Environment for configuration')
    parser.add_argument('--host', type=str, default=None,
                        help='RabbitMQ host (overrides configuration)')
    parser.add_argument('--port', type=int, default=None,
                        help='RabbitMQ port (overrides configuration)')
    parser.add_argument('--username', type=str, default=None,
                        help='RabbitMQ username (overrides configuration)')
    parser.add_argument('--password', type=str, default=None,
                        help='RabbitMQ password (overrides configuration)')
    parser.add_argument('--virtual-host', type=str, default=None,
                        help='RabbitMQ virtual host (overrides configuration)')
    parser.add_argument('--retry', action='store_true',
                        help='Retry connection on failure')
    parser.add_argument('--retry-count', type=int, default=5,
                        help='Number of connection retries')
    parser.add_argument('--retry-delay', type=int, default=5,
                        help='Delay between retries in seconds')
    
    return parser.parse_args()

def connect_to_rabbitmq(config, retry=False, retry_count=5, retry_delay=5):
    """
    Connect to RabbitMQ.
    
    Args:
        config: RabbitMQ configuration
        retry: Whether to retry on failure
        retry_count: Number of connection retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        Connection and channel tuple or None if failed
    """
    attempts = 1 if not retry else retry_count
    
    for attempt in range(1, attempts + 1):
        try:
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']} (attempt {attempt}/{attempts})")
            
            # Create connection parameters
            credentials = pika.PlainCredentials(
                username=config['username'],
                password=config['password']
            )
            
            connection_params = pika.ConnectionParameters(
                host=config['host'],
                port=config['port'],
                virtual_host=config['virtual_host'],
                credentials=credentials,
                connection_attempts=3,  # Local connection attempts
                retry_delay=2
            )
            
            # Connect to RabbitMQ
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            
            logger.info(f"Connected to RabbitMQ at {config['host']}:{config['port']}")
            return connection, channel
            
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            
            # Retry if specified
            if retry and attempt < attempts:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Connection failed, giving up")
                return None, None
        except Exception as e:
            logger.error(f"Unexpected error connecting to RabbitMQ: {e}")
            return None, None

def setup_exchanges_and_queues(channel):
    """
    Set up exchanges and queues.
    
    Args:
        channel: RabbitMQ channel
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create exchanges
        for exchange_key, exchange_config in EXCHANGES.items():
            config = get_exchange_config(exchange_key)
            
            logger.info(f"Declaring exchange: {config['name']} ({config['type']})")
            channel.exchange_declare(
                exchange=config['name'],
                exchange_type=config['type'],
                durable=config['durable'],
                auto_delete=config['auto_delete']
            )
        
        # Create queues
        for queue_key, queue_config in QUEUES.items():
            config = get_queue_config(queue_key)
            
            logger.info(f"Declaring queue: {config['name']}")
            channel.queue_declare(
                queue=config['name'],
                durable=config['durable'],
                exclusive=config['exclusive'],
                auto_delete=config['auto_delete'],
                arguments=config['arguments']
            )
            
            # Bind queue to exchange
            exchange_name = get_exchange_config('main')['name']
            routing_key = get_routing_key(queue_key)
            
            # For events queue, bind to events exchange
            if queue_key == 'events':
                exchange_name = get_exchange_config('events')['name']
                routing_key = get_routing_key('events')
            
            # For dead letter queue, bind to DLX exchange
            if queue_key == 'dead_letters':
                exchange_name = get_exchange_config('dlx')['name']
                routing_key = 'math_llm.dead_letters'
            
            logger.info(f"Binding queue {config['name']} to exchange {exchange_name} with routing key {routing_key}")
            channel.queue_bind(
                queue=config['name'],
                exchange=exchange_name,
                routing_key=routing_key
            )
        
        logger.info("All exchanges and queues have been set up successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up exchanges and queues: {e}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get RabbitMQ configuration
    config = get_rabbitmq_config(args.environment)
    
    # Override with command line arguments
    if args.host:
        config['host'] = args.host
    if args.port:
        config['port'] = args.port
    if args.username:
        config['username'] = args.username
    if args.password:
        config['password'] = args.password
    if args.virtual_host:
        config['virtual_host'] = args.virtual_host
    
    # Connect to RabbitMQ
    connection, channel = connect_to_rabbitmq(
        config,
        retry=args.retry,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay
    )
    
    if connection is None or channel is None:
        logger.error("Failed to connect to RabbitMQ")
        sys.exit(1)
    
    # Set up exchanges and queues
    success = setup_exchanges_and_queues(channel)
    
    # Close connection
    connection.close()
    
    # Exit with appropriate status code
    if success:
        logger.info("Message bus configuration initialized successfully")
        sys.exit(0)
    else:
        logger.error("Failed to initialize message bus configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
