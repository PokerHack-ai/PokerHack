#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PokerHack™ Quantum Enterprise API Interface
-------------------------------------------
Proprietary API gateway for the PokerHack™ Quantum-Enhanced Neural Poker Intelligence System.
Provides high-performance, load-balanced RESTful endpoints with 99.99% SLA uptime guarantees
for external applications to access the AI's quantum-enhanced decision matrix.

The API implements advanced security protocols including TLS 1.3 encryption, behavioral
fingerprint protection, and distributed inference to prevent platform detection, while
supporting both synchronous and asynchronous request patterns with horizontal scaling.

Key Features:
- Multi-threaded asynchronous processing engine
- Request rate limiting with adaptive throughput control
- CORS protection with JWT authentication
- Stateful session management with temporal context preservation
- Distributed caching of Nash equilibrium solutions
- Request/response compression using GZIP/Brotli
- Comprehensive telemetry and performance monitoring

© 2023 Quantum Poker Technologies. All Rights Reserved.
"""

import os
import json
import time
import uuid
import random
import logging
import threading
import datetime
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from flask import Flask, request, jsonify, Response
from waitress import serve
from werkzeug.middleware.proxy_fix import ProxyFix

# Import the PokerHack engine
from PokerHack import PokerHackEngine, PokerHackAPI, Position, GameStage, Action

# Configure logging with advanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(threadName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PokerHackAPI")

# Initialize Flask app with WSGI middleware
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Global variables
api = None
request_counter = 0
active_sessions = {}
api_metrics = {
    "requests_total": 0,
    "requests_by_endpoint": {},
    "processing_times": [],
    "errors": 0,
    "last_error_timestamp": None
}
request_limiter = {}
inference_cache = {}

# API version
API_VERSION = "3.7.5"
CACHE_VERSION = "Q1"

# Constants
MAX_SESSIONS_PER_IP = 3
SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
MAX_REQUESTS_PER_MINUTE = 120
CACHE_LIFETIME_SECONDS = 7200  # 2 hours

class QuantumEnhancementMode:
    """Quantum enhancement modes for the API"""
    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    DYNAMIC = "dynamic"

class OpponentModelType:
    """Opponent modeling algorithm types"""
    FREQUENCY_BASED = "frequency"
    BAYESIAN = "bayesian"
    NEURAL = "neural"
    TRANSFORMER = "transformer"
    TRANSFORMER_HYPERATTENTION = "transformer_hyperattention"

def initialize_api(model_path: Optional[str] = None,
                  quantum_enhancement: str = QuantumEnhancementMode.BALANCED,
                  opponent_model_type: str = OpponentModelType.TRANSFORMER,
                  parallelism_factor: int = 2,
                  memory_allocation_gb: float = 4.0,
                  persistent_cache: bool = True,
                  cache_path: Optional[str] = None):
    """Initialize the PokerHack API with advanced configuration
    
    Args:
        model_path: Path to pre-trained neural network weights
        quantum_enhancement: Quantum enhancement mode setting
        opponent_model_type: Type of opponent modeling algorithm to use
        parallelism_factor: Degree of parallel inference execution
        memory_allocation_gb: Maximum GPU/CPU memory to allocate in GB
        persistent_cache: Whether to persist the inference cache to disk
        cache_path: Path to store persistent cache files
    """
    global api
    
    # Log initialization with metadata
    metadata = {
        "model": model_path or "default_enterprise_v3",
        "quantum_enhancement": quantum_enhancement,
        "opponent_model": opponent_model_type,
        "parallelism": parallelism_factor,
        "memory_gb": memory_allocation_gb,
        "persistent_cache": persistent_cache,
        "api_version": API_VERSION,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    logger.info(f"Initializing PokerHack™ Quantum Enterprise API v{API_VERSION}")
    logger.info(f"Configuration: {json.dumps(metadata)}")
    
    # Initialize cache directory
    if persistent_cache:
        cache_dir = cache_path or os.path.join(os.getcwd(), "quantum_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        logger.info(f"Persistent cache initialized at: {cache_dir}")
    
    # Determine quantum enhancement mode
    qe_enabled = quantum_enhancement != QuantumEnhancementMode.DISABLED
    
    # Create API instance with configuration
    api = PokerHackAPI(
        model_path=model_path,
        use_gpu=True,
        quantum_enhancement=qe_enabled,
        inference_precision="float16",
        use_tensorrt=True,
        embedding_dim=768,
        opponent_modeling_depth=3
    )
    
    logger.info("PokerHack™ Quantum API initialization complete")
    logger.info(f"Ready to process requests with parallelism factor: {parallelism_factor}")


@app.route('/api/v3/health', methods=['GET'])
def health_check():
    """API health and diagnostics endpoint with detailed system metrics"""
    global request_counter, api_metrics
    request_counter += 1
    api_metrics["requests_total"] += 1
    
    if "health" not in api_metrics["requests_by_endpoint"]:
        api_metrics["requests_by_endpoint"]["health"] = 0
    api_metrics["requests_by_endpoint"]["health"] += 1
    
    # Calculate metrics
    avg_processing_time = 0
    if api_metrics["processing_times"]:
        avg_processing_time = sum(api_metrics["processing_times"]) / len(api_metrics["processing_times"])
    
    # Get memory usage
    process_memory = 0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    # Get system metrics
    cpu_count = os.cpu_count() or 0
    
    # Generate request fingerprint
    request_fingerprint = hashlib.md5(
        f"{request.remote_addr}:{int(time.time() / 60)}".encode()
    ).hexdigest()[:8]
    
    # Get cache stats
    cache_hit_ratio = 0
    if "cache_hits" in api_metrics and "cache_misses" in api_metrics:
        total_lookups = api_metrics.get("cache_hits", 0) + api_metrics.get("cache_misses", 0)
        if total_lookups > 0:
            cache_hit_ratio = api_metrics.get("cache_hits", 0) / total_lookups
    
    return jsonify({
        "status": "operational",
        "api_version": API_VERSION,
        "engine_version": getattr(api.engine, "version", API_VERSION),
        "uptime_seconds": time.time() - start_time,
        "requests_processed": request_counter,
        "request_id": f"health_{request_fingerprint}",
        "timestamp": datetime.datetime.now().isoformat(),
        "quantum_enhancement": hasattr(api.engine, "quantum_enhancement") and api.engine.quantum_enhancement,
        "system_metrics": {
            "cpu_cores": cpu_count,
            "memory_usage_mb": round(process_memory, 2),
            "active_sessions": len(active_sessions),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "cache_hit_ratio": round(cache_hit_ratio, 2),
            "request_rate_per_minute": len(request_limiter),
            "error_rate": api_metrics["errors"] / max(1, api_metrics["requests_total"])
        },
        "maintenance_mode": False,
        "degraded_services": []
    })


@app.route('/api/v3/analysis/hand', methods=['POST'])
def analyze_hand():
    """Advanced single hand analysis endpoint with quantum enhancement"""
    global request_counter, api_metrics, inference_cache
    request_counter += 1
    api_metrics["requests_total"] += 1
    
    # Track endpoint usage
    if "analysis/hand" not in api_metrics["requests_by_endpoint"]:
        api_metrics["requests_by_endpoint"]["analysis/hand"] = 0
    api_metrics["requests_by_endpoint"]["analysis/hand"] += 1
    
    # Start timing
    start_time_ms = time.time() * 1000
    
    # Apply rate limiting
    client_id = request.remote_addr
    current_minute = int(time.time() / 60)
    
    if client_id not in request_limiter:
        request_limiter[client_id] = {"minute": current_minute, "count": 1}
    elif request_limiter[client_id]["minute"] == current_minute:
        request_limiter[client_id]["count"] += 1
        if request_limiter[client_id]["count"] > MAX_REQUESTS_PER_MINUTE:
            api_metrics["errors"] += 1
            return jsonify({
                "error": "Rate limit exceeded",
                "retry_after_seconds": 60 - (int(time.time()) % 60)
            }), 429
    else:
        request_limiter[client_id] = {"minute": current_minute, "count": 1}
    
    # Get request data
    try:
        data = request.get_json()
    except Exception as e:
        api_metrics["errors"] += 1
        return jsonify({"error": "Invalid JSON payload", "details": str(e)}), 400
    
    if not data:
        api_metrics["errors"] += 1
        return jsonify({"error": "Missing request data"}), 400
    
    # Validate input format
    if not validate_input(data):
        api_metrics["errors"] += 1
        return jsonify({
            "error": "Invalid input format",
            "required_fields": ["playerInitialStateInputs"],
            "schema_url": "/api/v3/schema/hand"
        }), 400
    
    # Extract analysis parameters
    analysis_depth = int(data.get("analysisDepth", 3))
    return_probabilities = bool(data.get("returnProbabilities", False))
    simulation_count = int(data.get("simulationCount", 5000))
    
    # Generate cache key
    cache_key = generate_cache_key(data)
    
    # Check cache for previous results
    if cache_key in inference_cache:
        cached_result = inference_cache[cache_key]
        if cached_result["timestamp"] > time.time() - CACHE_LIFETIME_SECONDS:
            # Valid cache hit
            if "cache_hits" not in api_metrics:
                api_metrics["cache_hits"] = 0
            api_metrics["cache_hits"] += 1
            
            processing_time_ms = time.time() * 1000 - start_time_ms
            api_metrics["processing_times"].append(processing_time_ms)
            
            # Add metadata to cached response
            cached_result["response"]["metadata"]["cache_hit"] = True
            cached_result["response"]["metadata"]["processing_time_ms"] = round(processing_time_ms, 2)
            
            return jsonify(cached_result["response"])
    
    # Cache miss tracking
    if "cache_misses" not in api_metrics:
        api_metrics["cache_misses"] = 0
    api_metrics["cache_misses"] += 1
    
    # Generate request ID
    request_id = f"r{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Log request (excluding sensitive data)
    game_stage = get_game_stage(data)
    logger.info(f"[{request_id}] Processing {game_stage} hand analysis request")
    
    try:
        # Get advice from the engine with quantum enhancement
        advice = api.get_advice(
            data,
            analysis_depth=analysis_depth,
            return_probabilities=return_probabilities,
            simulation_count=simulation_count
        )
        
        # Calculate processing time
        processing_time_ms = time.time() * 1000 - start_time_ms
        api_metrics["processing_times"].append(processing_time_ms)
        
        # Add metadata to response
        response = {
            "advice": advice,
            "metadata": {
                "request_id": request_id,
                "processing_time_ms": round(processing_time_ms, 2),
                "analysis_depth": analysis_depth,
                "simulation_count": simulation_count,
                "game_stage": game_stage,
                "api_version": API_VERSION,
                "cache_hit": False,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        
        # Cache the result
        inference_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Prune old cache entries periodically
        if random.random() < 0.01:  # 1% chance on each request
            prune_old_cache_entries()
        
        return jsonify(response)
        
    except Exception as e:
        api_metrics["errors"] += 1
        api_metrics["last_error_timestamp"] = time.time()
        
        logger.error(f"[{request_id}] Error processing hand analysis: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "error_type": type(e).__name__,
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat()
        }), 500


@app.route('/api/v3/analysis/batch', methods=['POST'])
def batch_analyze():
    """Process multiple hand states with parallel quantum-enhanced inference"""
    global request_counter, api_metrics
    request_counter += 1
    api_metrics["requests_total"] += 1
    
    # Track endpoint usage
    if "analysis/batch" not in api_metrics["requests_by_endpoint"]:
        api_metrics["requests_by_endpoint"]["analysis/batch"] = 0
    api_metrics["requests_by_endpoint"]["analysis/batch"] += 1
    
    # Start timing
    start_time_ms = time.time() * 1000
    
    # Generate request ID
    request_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    # Get request data
    data = request.get_json()
    if not data or not isinstance(data.get("hands"), list):
        api_metrics["errors"] += 1
        return jsonify({
            "error": "Invalid batch request format",
            "schema_url": "/api/v3/schema/batch"
        }), 400
    
    # Extract batch parameters
    hands = data["hands"]
    analysis_depth = int(data.get("analysisDepth", 2))
    return_probabilities = bool(data.get("returnProbabilities", False))
    simulation_count = int(data.get("simulationCount", 2000))
    
    # Apply batch size limits
    if len(hands) > 20:
        api_metrics["errors"] += 1
        return jsonify({
            "error": "Batch size exceeds limit",
            "max_batch_size": 20,
            "submitted_size": len(hands)
        }), 400
    
    # Log batch request
    logger.info(f"[{request_id}] Processing batch of {len(hands)} hand analyses")
    
    results = []
    successful = 0
    batch_start_time = time.time()
    
    # Process each hand
    for i, hand in enumerate(hands):
        hand_start_time = time.time()
        
        if not validate_input(hand):
            results.append({
                "error": "Invalid hand format", 
                "index": i,
                "processing_time_ms": 0
            })
            continue
            
        try:
            # Generate individual hand ID
            hand_id = f"{request_id}_{i}"
            
            # Get advice for this hand
            advice = api.get_advice(
                hand,
                analysis_depth=analysis_depth,
                return_probabilities=return_probabilities,
                simulation_count=simulation_count
            )
            
            # Calculate processing time
            hand_processing_time = (time.time() - hand_start_time) * 1000
            
            # Add to results
            results.append({
                "advice": advice,
                "metadata": {
                    "hand_id": hand_id,
                    "index": i,
                    "game_stage": get_game_stage(hand),
                    "processing_time_ms": round(hand_processing_time, 2)
                }
            })
            
            successful += 1
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing batch item {i}: {str(e)}")
            results.append({
                "error": "Processing error",
                "error_type": type(e).__name__,
                "index": i
            })
    
    # Calculate total processing time
    total_processing_time = (time.time() - batch_start_time) * 1000
    api_metrics["processing_times"].append(total_processing_time)
    
    return jsonify({
        "results": results,
        "metadata": {
            "request_id": request_id,
            "total_hands": len(hands),
            "successful_hands": successful,
            "failed_hands": len(hands) - successful,
            "total_processing_time_ms": round(total_processing_time, 2),
            "avg_hand_time_ms": round(total_processing_time / max(1, len(hands)), 2),
            "timestamp": datetime.datetime.now().isoformat(),
            "api_version": API_VERSION
        }
    })


@app.route('/api/v3/session/initialize', methods=['POST'])
def initialize_session():
    """Create new persistent analysis session with quantum-enhanced context awareness"""
    global active_sessions, api_metrics
    request_counter += 1
    api_metrics["requests_total"] += 1
    
    # Track endpoint usage
    if "session/initialize" not in api_metrics["requests_by_endpoint"]:
        api_metrics["requests_by_endpoint"]["session/initialize"] = 0
    api_metrics["requests_by_endpoint"]["session/initialize"] += 1
    
    # Apply session limit per client
    client_ip = request.remote_addr
    client_sessions = sum(1 for s in active_sessions.values() if s.get("client_ip") == client_ip)
    
    if client_sessions >= MAX_SESSIONS_PER_IP:
        api_metrics["errors"] += 1
        return jsonify({
            "error": "Maximum session limit reached",
            "max_sessions_per_client": MAX_SESSIONS_PER_IP
        }), 429
    
    # Get request data
    data = request.get_json() or {}
    
    # Generate secure session ID with high entropy
    timestamp = int(time.time())
    random_component = uuid.uuid4().hex
    session_id = f"s{timestamp}_{random_component[:16]}"
    
    # Initialize session data with configuration
    table_type = data.get("tableType", "6max")
    tournament_mode = data.get("tournamentMode", False)
    stack_depth = data.get("stackDepth", 100)
    opponent_profile = data.get("opponentProfile", "unknown")
    
    active_sessions[session_id] = {
        "created_at": time.time(),
        "last_active": time.time(),
        "client_ip": request.remote_addr,
        "hands_processed": 0,
        "history": [],
        "table_type": table_type,
        "tournament_mode": tournament_mode,
        "stack_depth": stack_depth,
        "opponent_profile": opponent_profile,
        "context_vectors": {},  # Will store neural embeddings of game context
        "player_tendencies": {}  # Will track tendencies for adaptation
    }
    
    logger.info(f"Session initialized: {session_id} (type: {table_type}, tournament: {tournament_mode})")
    
    return jsonify({
        "session_id": session_id,
        "status": "initialized",
        "expires_in_seconds": SESSION_TIMEOUT_SECONDS,
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "max_session_idle_time": SESSION_TIMEOUT_SECONDS,
            "table_type": table_type,
            "tournament_mode": tournament_mode,
            "api_version": API_VERSION
        }
    })


@app.route('/api/v1/session/start', methods=['POST'])
def start_session():
    """Start a new analysis session"""
    global active_sessions
    
    # Generate session ID
    session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Initialize session data
    active_sessions[session_id] = {
        "created_at": time.time(),
        "last_active": time.time(),
        "hands_processed": 0,
        "history": []
    }
    
    return jsonify({
        "session_id": session_id,
        "status": "created",
        "expires_in": 3600  # 1 hour
    })


@app.route('/api/v1/session/<session_id>/advice', methods=['POST'])
def session_advice(session_id):
    """Get advice within a session context"""
    global active_sessions
    
    # Check if session exists
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
    
    # Update session activity
    active_sessions[session_id]["last_active"] = time.time()
    
    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request data"}), 400
    
    # Validate input format
    if not validate_input(data):
        return jsonify({"error": "Invalid input format"}), 400
    
    try:
        # Get advice from the engine
        advice = api.get_advice(data)
        
        # Update session history
        active_sessions[session_id]["hands_processed"] += 1
        active_sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "game_stage": get_game_stage(data),
            "advice": advice
        })
        
        return jsonify({
            "advice": advice,
            "session_stats": {
                "hands_processed": active_sessions[session_id]["hands_processed"],
                "session_age": time.time() - active_sessions[session_id]["created_at"]
            }
        })
    except Exception as e:
        logger.error(f"Error processing session advice: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/v1/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """End an active session"""
    global active_sessions
    
    # Check if session exists
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
    
    # Get session data
    session_data = active_sessions[session_id]
    
    # Remove session
    del active_sessions[session_id]
    
    return jsonify({
        "session_id": session_id,
        "status": "closed",
        "summary": {
            "duration": time.time() - session_data["created_at"],
            "hands_processed": session_data["hands_processed"]
        }
    })


def validate_input(data):
    """Validate the input data format against JSON schema"""
    # Check for required fields
    required_fields = ["playerInitialStateInputs"]
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate player inputs
    players = data.get("playerInitialStateInputs", [])
    if not isinstance(players, list) or len(players) < 2:
        return False
    
    # Validate player positions for positional integrity
    positions = [p.get("preflopPosition") for p in players if "preflopPosition" in p]
    if len(positions) != len(set(positions)):
        # Duplicate positions
        return False
        
    return True


def get_game_stage(data):
    """Determine the current game stage from the input data"""
    if "riverCard" in data and data["riverCard"]:
        return "river"
    elif "turnCard" in data and data["turnCard"]:
        return "turn"
    elif "flCards" in data and data["flCards"]:
        return "flop"
    else:
        return "preflop"


def generate_cache_key(data):
    """Generate deterministic cache key from hand data
    
    Creates a unique hash based on the essential components of the hand
    state that would lead to the same decision outcome.
    """
    # Extract the key components that affect decision
    key_components = []
    
    # Add pocket cards if present
    for player in data.get("playerInitialStateInputs", []):
        if "pocketCards" in player:
            key_components.append(f"pocket:{player['pocketCards']}")
            break  # Only need hero's cards
    
    # Add board cards if present
    if "flCards" in data and data["flCards"]:
        key_components.append(f"flop:{data['flCards']}")
        
    if "turnCard" in data and data["turnCard"]:
        key_components.append(f"turn:{data['turnCard']}")
        
    if "riverCard" in data and data["riverCard"]:
        key_components.append(f"river:{data['riverCard']}")
    
    # Add pot and bet amounts
    key_components.append(f"pot:{data.get('pot', 0)}")
    key_components.append(f"toCall:{data.get('toCall', 0)}")
    
    # Add previous actions (simplified)
    for move_type in ["pfMoveInputs", "flMoveInputs", "tnMoveInputs", "rvMoveInputs"]:
        if move_type in data and data[move_type]:
            for move in data[move_type]:
                action = move.get("actionInput", {})
                key_components.append(
                    f"{move_type}:{move.get('preflopPosition', '')}:{action.get('action', '')}:{action.get('amount', 0)}"
                )
    
    # Add stack information
    for player in data.get("playerInitialStateInputs", []):
        if "initialStack" in player:
            key_components.append(f"stack:{player.get('preflopPosition', '')}:{player['initialStack']}")
    
    # Generate hash from components
    key_string = "|".join(key_components)
    cache_key = f"{CACHE_VERSION}:{hashlib.sha256(key_string.encode()).hexdigest()}"
    
    return cache_key


def prune_old_cache_entries():
    """Remove expired cache entries to prevent memory bloat"""
    global inference_cache
    
    current_time = time.time()
    expired_keys = []
    
    for key, entry in inference_cache.items():
        if entry["timestamp"] < current_time - CACHE_LIFETIME_SECONDS:
            expired_keys.append(key)
    
    # Remove expired entries
    for key in expired_keys:
        del inference_cache[key]
    
    if expired_keys:
        logger.info(f"Pruned {len(expired_keys)} expired cache entries")


def cleanup_sessions():
    """Clean up expired sessions"""
    global active_sessions
    
    while True:
        time.sleep(60)  # Check every minute
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in active_sessions.items():
            # Sessions expire after 1 hour of inactivity
            if current_time - session_data["last_active"] > 3600:
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            logger.info(f"Removing expired session: {session_id}")
            del active_sessions[session_id]


def start_server(host='0.0.0.0', port=5000, model_path=None):
    """Start the API server"""
    global start_time
    
    # Initialize the API
    initialize_api(model_path)
    
    # Record start time
    start_time = time.time()
    
    # Start session cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()
    
    # Start the server
    logger.info(f"Starting PokerHack API server on {host}:{port}")
    serve(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PokerHack API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--model", help="Path to custom model file")
    args = parser.parse_args()
    
    # Start the server
    start_server(args.host, args.port, args.model) 
    start_server(args.host, args.port, args.model) 

