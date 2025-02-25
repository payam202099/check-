
import hashlib
import json
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import requests
from flask import Flask, jsonify, request

    def register_node(self, address):
        """
        Add a new node to the list of nodes
        :param address: Address of node. Eg. 'http://192.168.0.5:5000'
        """
        parsed_url = urlparse(address)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')
            
    def valid_chain(self, chain):
        """
        Determine if a given blockchain is valid
        :param chain: A blockchain
        :return: True if valid, False if not
        """
        last_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False
            if not self.valid_proof(last_block['proof'], block['proof'], last_block_hash):
                return False
            last_block = block
            current_index += 1
        return True

    
    def new_block(self, proof, previous_hash):
        """
        Create a new Block in the Blockchain
        :param proof: The proof given by the Proof of Work algorithm
        :param previous_hash: Hash of previous Block
        :return: New Block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    @staticmethod
    
    def proof_of_work(self,last_block):
        """
        Simple Proof of Work Algorithm:
        - Find a number p' such that hash(pp') contains leading 4 zeroes, where p is the previous p'
        - p is the previous proof, and p' is the new proof
        :param last_block: <dict> last Block
        :return: <int>
        """
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)
        proof = 0
        while self.valid_proof(last_proof,proof,last_hash) is False:
            proof += 1
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof,last_hash):
        """
        Validates the proof
        :param last_proof: Previous Proof
        :param proof: Current Proof
        :param last_hash: The hash of the Previous Block
        :return: <bool> True if correct, False if not.
        """
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
        
    def adjust_difficulty(self):
            if len(self.chain) < 2:
                return self.difficulty
            
            time_taken = self.chain[-1]['timestamp'] - self.chain[-2]['timestamp']
            if time_taken < self.block_time / 2:
                self.difficulty += 1
            elif time_taken > self.block_time * 2 and self.difficulty > 1 :
                self.difficulty -= 1
            return self.difficulty
            
    def mine(self, miner_address):
        self.add_reward_transaction(miner_address)
        last_block = self.last_block
        proof = self.proof_of_work(last_block)
        previous_hash = self.hash(last_block)
        block = self.new_block(proof, previous_hash)
        self.adjust_difficulty()
        return block
    
    def get_balance(self, address):
            """Calculates the balance for a given address"""
            balance = 0
            for block in self.chain:
                for transaction in block['transactions']:
                    if transaction['sender'] == address:
                        balance -= transaction['amount'] + transaction.get('fee',0)
                    if transaction['recipient'] == address:
                        balance += transaction['amount']
            return balance

app = Flask(__name__)
node_identifier = str(uuid4()).replace('-', '')
blockchain = Blockchain()

@app.route('/mine', methods=['POST'])
def mine():
    values = request.get_json()
    required = ['miner_address']
    if not all(k in values for k in required):
        return 'Missing values', 400
    miner_address = values['miner_address']
    block = blockchain.mine(miner_address)
    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400
    sender = values['sender']
    recipient = values['recipient']
    amount = values['amount']
    fee = values.get('fee',0)
    index = blockchain.new_transaction(sender, recipient, amount, fee)
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400
    for node in nodes:
        blockchain.register_node(node)
    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }
    return jsonify(response), 200

@app.route('/balance/<address>', methods=['GET'])
def get_balance(address):
    balance = blockchain.get_balance(address)
    response = {'balance':balance}
    return jsonify(response),200

@app.route('/nodes/view',methods=['GET'])
def view_nodes():
    response = {'nodes': list(blockchain.nodes)}
    return jsonify(response),200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=random.randint(5000, 8000))
