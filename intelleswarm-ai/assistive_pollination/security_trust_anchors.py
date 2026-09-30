# Security and Trust Anchors (Cryptography)
# File: security_trust_anchors.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
#
# Simulates the cryptographic protocols used to establish trust between the
# drone, the swarm coordinator, and the cloud base. Ensures command authenticity
# (Trust Anchor) and data integrity (HMAC concept).

import hashlib
import json
import random
import time
import hmac
from typing import Dict, Any

# --- Configuration Constants (Simulated Secrets) ---
# A strong, shared secret key known only to the verified Swarm Coordinator and the drone.
# In a real system, this would be a hardware-secured private key (Trust Anchor).
SWARM_SECRET_KEY = b"IntelliSwarm_Drone_739_Secure_Key_2025!"
DRONE_ID = 7


# --- Core Cryptographic Logic ---

class TrustAnchor:
    """Handles secure signing and verification using a shared secret (HMAC equivalent)."""

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def _serialize_command(self, command_data: Dict[str, Any]) -> bytes:
        """Converts command data to a consistent, canonical JSON string for hashing."""
        # Use sorted keys to ensure the output is deterministic
        return json.dumps(command_data, sort_keys=True).encode('utf-8')

    def sign_command(self, command_data: Dict[str, Any]) -> str:
        """
        Generates a Message Authentication Code (MAC) for command integrity.
        Simulates the digital signature process using HMAC-SHA256.
        """
        serialized_data = self._serialize_command(command_data)

        # Use HMAC for integrity check based on the shared secret
        signature = hmac.new(self.secret_key, serialized_data, hashlib.sha256).hexdigest()

        return signature

    def verify_command(self, command_data: Dict[str, Any], received_signature: str) -> bool:
        """
        Verifies the command integrity and authenticity against the expected signature.
        :returns: True if the command is authentic and untampered, False otherwise.
        """
        expected_signature = self.sign_command(command_data)

        # Use a constant time comparison to avoid timing attacks
        return hmac.compare_digest(expected_signature, received_signature)


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    crypto_engine = TrustAnchor(SWARM_SECRET_KEY)

    print("--- Security and Trust Anchors (Cryptography) Simulation ---")
    print(f"Simulating drone: {DRONE_ID}")
    print("-" * 60)

    # --- SCENARIO 1: AUTHENTIC AND UNTAMPERED COMMAND ---

    # 1. Swarm Coordinator (Server) generates a critical command
    authentic_command = {
        "source": "Swarm_Coordinator",
        "command": "RTH_ALL",
        "target": "All",
        "timestamp": time.time(),
        "mission_id": "M-Alpha-42"
    }

    # 2. Server signs the command
    authentic_signature = crypto_engine.sign_command(authentic_command)
    print(f"Server Signed Command (RTH_ALL): {authentic_signature[:16]}...")

    # 3. Drone receives the command and verifies the signature
    is_verified_authentic = crypto_engine.verify_command(authentic_command, authentic_signature)

    print(f"Drone Verification Result (Authentic): {is_verified_authentic}")
    if is_verified_authentic:
        print("[ACTION] Drone executes RTH_ALL command.")

    print("-" * 60)

    # --- SCENARIO 2: MAN-IN-THE-MIDDLE ATTACK (Tampering) ---

    # 1. An attacker intercepts the authentic command
    tampered_command = authentic_command.copy()

    # 2. The attacker changes the command to an unauthorized action (e.g., ABORT_MISSION)
    tampered_command["command"] = "ABORT_MISSION"

    # 3. Drone receives the tampered command and the original authentic signature
    # Note: The signature does not match the tampered data, as the data changed.
    is_verified_tampered = crypto_engine.verify_command(tampered_command, authentic_signature)

    print(f"Attacker Tampered Command: {tampered_command['command']}")
    print(f"Drone Verification Result (Tampered): {is_verified_tampered}")
    if not is_verified_tampered:
        print("[ACTION] Drone REJECTS command. Integrity check failed.")

    print("-" * 60)

    # --- SCENARIO 3: REPLAY ATTACK (Altering Timestamp) ---

    # 1. Attacker tries to replay an old command by changing the timestamp (data change)
    replay_command = authentic_command.copy()
    replay_command["timestamp"] = time.time() + 999999.0  # Future timestamp

    # 2. Drone receives the replayed command with the original signature
    is_verified_replay = crypto_engine.verify_command(replay_command, authentic_signature)

    print(f"Attacker Replay Command: {replay_command['timestamp']:.2f}")
    print(f"Drone Verification Result (Replay Attempt): {is_verified_replay}")

    # Final Validation
    print("\n--- MVP Validation Check ---")
    if is_verified_authentic and not is_verified_tampered:
        print(
            "MVP Validation: Successful. The cryptographic Trust Anchor correctly verified the authentic command and "
            "successfully rejected the tampered command, ensuring command integrity and swarm security.")
    else:
        print("MVP Validation: Failure. The signature verification logic did not perform as expected.")
