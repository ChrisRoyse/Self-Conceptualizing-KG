"""
Neo4j client module for handling database connections and operations.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Neo4j database client with updated connection handling for Neo4j 5.26.4"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", max_attempts: int = 3):
        """Initialize Neo4j client with connection parameters"""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_attempts = max_attempts
        self.driver = None
        self._connect()
        self._verify_plugins()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database with retry logic"""
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"\nConnection attempt {attempt}/{self.max_attempts}")
                logger.info(f"URI: {self.uri}")
                logger.info(f"Username: {self.user}")
                
                # Configure driver with Neo4j 5.x compatible settings
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    # Connection settings tuned for Neo4j 5.x
                    connection_timeout=30,
                    max_transaction_retry_time=30000,
                    connection_acquisition_timeout=60,
                    # Match browser configuration
                    user_agent="neo4j-browser_v5.15.0"
                )
                
                # Test basic connectivity
                logger.info("Testing connectivity...")
                self.driver.verify_connectivity()
                
                # Verify connection with simple query
                with self.driver.session(database=self.database) as session:
                    result = session.run("RETURN 1 AS test")
                    if result.single()["test"] == 1:
                        logger.info(f"Successfully connected to Neo4j database at {self.uri}")
                        return
                    
            except AuthError as auth_error:
                logger.error(f"Authentication error: {str(auth_error)}")
                if "rate limit" in str(auth_error).lower():
                    logger.info("Rate limit hit - waiting before retry...")
                elif "unauthorized" in str(auth_error).lower():
                    logger.error("\nPossible solutions:")
                    logger.error("1. Verify password is correct")
                    logger.error("2. Try resetting password in Neo4j Browser:")
                    logger.error("   ALTER USER neo4j SET PASSWORD 'NewPassword123'")
                
            except ServiceUnavailable as e:
                logger.error(f"Neo4j service unavailable: {str(e)}")
                logger.error("Check if Neo4j is running in Neo4j Desktop")
                
            except Exception as e:
                logger.error(f"Connection error: {str(e)}")
                if hasattr(e, 'code'):
                    logger.error(f"Error code: {e.code}")
            
            # Wait before retry unless it's the last attempt
            if attempt < self.max_attempts:
                wait_time = 10
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        raise Exception("Failed to connect to Neo4j after multiple attempts")
    
    def _verify_plugins(self) -> None:
        """Verify that required plugins (APOC, GDS) are available"""
        try:
            with self.driver.session(database=self.database) as session:
                # Check APOC
                result = session.run("CALL apoc.help('apoc')")
                result.consume()
                logger.info("APOC plugin is available")
                
                # Check GDS
                result = session.run("CALL gds.list()")
                result.consume()
                logger.info("Graph Data Science library is available")
        except Exception as e:
            logger.warning(f"Plugin verification warning: {str(e)}")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Neo4jError as e:
            logger.error(f"Neo4j query error: {str(e)}")
            if e.code == "Neo.ClientError.Security.Unauthorized":
                logger.error("Authentication failed - please check credentials")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def execute_gds_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Graph Data Science query and return results"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Neo4jError as e:
            logger.error(f"GDS query error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error executing GDS query: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed") 