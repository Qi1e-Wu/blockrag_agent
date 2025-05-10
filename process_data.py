import os
import json
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
from decimal import Decimal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_wei_to_eth(wei_value):
    """Convert Wei to ETH"""
    try:
        return str(Decimal(wei_value) / Decimal(10**18))
    except:
        return wei_value

def format_timestamp(timestamp):
    """Format timestamp to readable date"""
    try:
        if isinstance(timestamp, str):
            return timestamp
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)

def create_documents_dir():
    """Create document directory structure"""
    base_dir = Path("documents")
    normal_dir = base_dir / "normal"
    malicious_dir = base_dir / "malicious"
    
    # Create directories
    for dir_path in [base_dir, normal_dir, malicious_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return normal_dir, malicious_dir

def process_transaction_data(tx_hash, tx_data, is_malicious=False):
    """Process single transaction data with enhanced JSON handling"""
    try:
        # Extract key information with better error handling
        processed_data = {
            "tx_hash": tx_hash,
            "from_address": tx_data.get("from", ""),
            "to_address": tx_data.get("to", ""),
            "value": format_wei_to_eth(tx_data.get("value", "0")),
            "gas": tx_data.get("gas", "0"),
            "gas_used": tx_data.get("gasUsed", "0"),
            "type": tx_data.get("type", ""),
            "func": tx_data.get("func", ""),
            "is_malicious": is_malicious
        }

        # Process state changes if available
        state_changes = []
        if "state" in tx_data:
            for state in tx_data["state"]:
                state_changes.append({
                    "type": state.get("type", ""),
                    "address": state.get("address", ""),
                    "key": state.get("key", ""),
                    "value": state.get("value", "")
                })
        
        # Process arguments if available
        args = []
        if "args" in tx_data:
            for arg in tx_data["args"]:
                args.append({
                    "type": arg.get("type", ""),
                    "data": arg.get("data", "")
                })
        
        # Convert to text format with better organization
        text = f"""Transaction Details:
------------------
Transaction Hash: {processed_data['tx_hash']}
Transaction Type: {processed_data['type']}

Addresses:
----------
From: {processed_data['from_address']}
To: {processed_data['to_address']}

Transaction Parameters:
---------------------
Value (ETH): {processed_data['value']}
Gas Limit: {processed_data['gas']}
Gas Used: {processed_data['gas_used']}
Function Signature: {processed_data['func']}

Arguments:
----------
"""
        # Add arguments
        for idx, arg in enumerate(args):
            text += f"Arg {idx + 1}:\n"
            text += f"  Type: {arg['type']}\n"
            text += f"  Data: {arg['data']}\n"

        # Add state changes
        text += "\nState Changes:\n-------------\n"
        for idx, state in enumerate(state_changes):
            text += f"Change {idx + 1}:\n"
            text += f"  Type: {state['type']}\n"
            text += f"  Address: {state['address']}\n"
            text += f"  Key: {state['key']}\n"
            text += f"  Value: {state['value']}\n"

        text += f"\nStatus:\n-------\nIs Malicious: {'Yes' if is_malicious else 'No'}\n"
        return text
    except Exception as e:
        logger.error(f"Error processing transaction data: {str(e)}")
        return f"Error processing transaction: {str(e)}"

def process_data_files():
    """Process all data files with enhanced error handling"""
    normal_dir, malicious_dir = create_documents_dir()
    data_dir = Path("data")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Process each project's data
    for project_dir in data_dir.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith('.'):
            continue
            
        logger.info(f"Processing project: {project_dir.name}")
        
        # Process normal transactions
        normal_txs_dir = project_dir / "benign_txs"
        if normal_txs_dir.exists():
            for tx_file in normal_txs_dir.glob("*.json"):
                try:
                    with open(tx_file, 'r') as f:
                        tx_data = json.load(f)
                    
                    # Process each transaction in the file
                    for tx_hash, tx_list in tx_data.items():
                        for idx, tx in enumerate(tx_list):
                            processed_text = process_transaction_data(tx_hash, tx, is_malicious=False)
                            output_file = normal_dir / f"{project_dir.name}_{tx_file.stem}_{idx}.txt"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(processed_text)
                except Exception as e:
                    logger.error(f"Error processing file {tx_file}: {str(e)}")
        
        # Process malicious transactions
        malicious_txs_dir = project_dir / "malicious_txs"
        if malicious_txs_dir.exists():
            for tx_file in malicious_txs_dir.glob("*.json"):
                try:
                    with open(tx_file, 'r') as f:
                        tx_data = json.load(f)
                    
                    # Process each transaction in the file
                    for tx_hash, tx_list in tx_data.items():
                        for idx, tx in enumerate(tx_list):
                            processed_text = process_transaction_data(tx_hash, tx, is_malicious=True)
                            output_file = malicious_dir / f"{project_dir.name}_{tx_file.stem}_{idx}.txt"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(processed_text)
                except Exception as e:
                    logger.error(f"Error processing file {tx_file}: {str(e)}")

def main():
    logger.info("Starting blockchain transaction data processing...")
    process_data_files()
    logger.info("Data processing completed!")
    logger.info("Processed files have been saved in the documents directory")
    logger.info("- normal directory contains normal transaction data")
    logger.info("- malicious directory contains malicious transaction data")

if __name__ == "__main__":
    main() 