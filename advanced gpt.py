from calendar import EPOCH
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
    
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPClassifier
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import requests
import newspaper
from googlesearch import search
import json
import random
import numpy as np
import re
from collections import Counter
import logging
import time
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import google.cloud.storage as gcs
    from google.cloud import bigquery
    from google.auth import credentials
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    print("Warning: Google Cloud or AWS libraries not found, some functions may be disabled")
    gcs = None
    bigquery = None
    credentials = None
    boto3 = None
    NoCredentialsError = Exception  # dummy class for error handling


# Other libraries for web scraping

from bs4 import BeautifulSoup

# Libraries for knowledge representation and reasoning
try:
    import networkx as nx
    import rdflib
except ImportError:
    print("Warning: Knowledge graph libraries not found, some functions may be disabled")
    nx = None
    rdflib = None
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
NUM_WORKERS = os.cpu_count()  # Utilize maximum CPU cores
MAX_VOCAB_SIZE = 10000000   # Large Vocabulary
PATIENCE = 50 #Early Stopping
LOG_DIR = 'tensorboard_logs/'
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Define constants for different aspects of training
MAX_SEQ_LEN = 16384 # Maximum sequence length we can process
LARGE_BATCH_SIZE = 4096 # Use large batch size if possible, and allow for it to be adjusted dynamically
LEARNING_RATE = 1e-4 # initial learning rate
NUM_LAYERS = 500# more layers for more complex processing
NUM_HEADS = 19  # More heads for attention
EMBEDDING_DIM = 4096 # dimensionality for representation
HIDDEN_DIM = 16384 # hidden layers size
DROPOUT = 0.1 # for regularization
MIN_FREQ = 5 # frequency required for a token
NUM_EPOCHS = 200000000 # For training
EVAL_STEPS = 500 # evaluate every number of steps
SAVE_STEPS = 5000000  # save model every number of steps
GRADIENT_CLIPPING = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set to GPU if available
LARGE_BATCH_SIZE = 16384
# Advanced Features
GLOBAL_GRAD_ACCUMULATION = 4 # Accumulate gradients for batch size
WEIGHT_DECAY = 1e-5  # L2 regularization
WARMUP_STEPS = 1000  # Linear learning rate warmup
ADAM_EPSILON = 1e-8 # Smoothing terms of adam optimizer
MAX_GRAD_NORM = 3.0 # Gradient clipping to prevent explosion
LOGGING_INTERVAL = 50 # Log metrics on every few steps

# TPU config if any, adjust according to TPU env setup

def advanced_text_preprocessing(text: str, lemmatize:bool = True) -> str:
    """Performs advanced text preprocessing, including stemming and lemmatization"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)
# --- Data Retrieval and Processing Functions ---
def fetch_data_from_gcp(bucket_name: str, blob_name: str, credentials_path: str) -> str:
    """
    Fetches data from Google Cloud Storage bucket.
    Args:
        bucket_name: Name of the GCP bucket.
        blob_name: Name of the blob (file).
        credentials_path: Path to the GCP credentials file.
    Returns:
        Data content as a string.
    """
    if gcs is None:
        logging.warning("Google Cloud Storage is not available.")
        return ""
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_text()
        logging.info(f"Successfully downloaded {blob_name} from GCS")
        return data
    except Exception as e:
        logging.error(f"Error fetching data from GCP: {e}")
        return ""


def fetch_data_from_aws(bucket_name: str, file_name: str) -> str:
    """
        Fetches data from AWS S3 bucket.
        Args:
            bucket_name: Name of the AWS bucket.
            file_name: Name of the file.
        Returns:
           Data content as a string.
    """
    if boto3 is None:
        logging.warning("AWS S3 is not available.")
        return ""
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        data = response['Body'].read().decode('utf-8')
        logging.info(f"Successfully downloaded {file_name} from AWS S3")
        return data
    except NoCredentialsError:
        logging.error("AWS credentials not found. Please configure AWS CLI or environment variables")
        return ""
    except Exception as e:
        logging.error(f"Error fetching data from AWS S3: {e}")
        return ""


def fetch_data_from_url(url: str) -> str:
    """
       Fetches data from a url.
       Args:
           url: url to fetch data from
       Returns:
            Data content as a string
    """
    try:
        response = requests.get(url, timeout=10) # set a timeout
        response.raise_for_status() # raise exception for bad status code
        text = response.text
        logging.info(f"Successfully downloaded content from {url}")
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {url} {e}")
        return ""

def preprocess_text(text: str) -> str:
    """
    Preprocesses a given text by lower casing and removing special characters.
    Args:
         text : The input text.
    Returns:
         The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_json_data(data_path: str) -> List[Dict]:
    """
    Loads JSON data from a file.
    Args:
         data_path: The path to the JSON file
    Returns:
         A list of JSON objects.
    """
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully loaded JSON data from {data_path}")
            return data
    except Exception as e:
         logging.error(f"Error loading JSON data {e}")
         return []
def extract_pairs_from_json(json_data: List[Dict]) -> List[Tuple[str, str]]:
    """
    Extract question-answer pairs from the json data
    Args:
        json_data: List of json objects with keys "question" and "answer"
    Returns:
        A list of pairs as tuples.
    """
    pairs = []
    for item in json_data:
        question = item.get('question')
        answer = item.get('answer')
        if question and answer:
            pairs.append((preprocess_text(question), preprocess_text(answer)))
    logging.info(f"Successfully extracted {len(pairs)} question-answer pairs")
    return pairs



def load_and_preprocess_data(data_path: str, data_type: str = "file") -> List[Tuple[str, str]]:
    """
    Loads, preprocesses, and returns text data from a file or a URL.
    Args:
         data_path : The path or URL to load from
         data_type: File or URL or JSON
    Returns:
         A list of tuples as question-answer pairs.
    """
    try:
        if data_type == "file":
            json_data = load_json_data(data_path)
            return extract_pairs_from_json(json_data)
        elif data_type == "url":
            text = fetch_data_from_url(data_path)
            pairs = []
            if text:
                pairs.append(("question_from_url", text)) # Treat the fetched text as an answer to a generic question
            return pairs
        elif data_type == "gcp":
            parts = data_path.split("||")
            if len(parts) != 3:
                logging.error("Invalid data path format for GCP, expecting bucket||blob||cred_path")
                return []
            bucket_name, blob_name, credentials_path = parts
            text = fetch_data_from_gcp(bucket_name, blob_name, credentials_path)
            pairs = []
            if text:
                 pairs.append(("question_from_gcp", text)) # Treat the fetched text as an answer to a generic question
            return pairs
        elif data_type == "aws":
            parts = data_path.split("||")
            if len(parts) != 2:
                logging.error("Invalid data path format for AWS, expecting bucket||file_name")
                return []
            bucket_name, file_name = parts
            text = fetch_data_from_aws(bucket_name, file_name)
            pairs = []
            if text:
                 pairs.append(("question_from_aws", text)) # Treat the fetched text as an answer to a generic question
            return pairs
        else:
            logging.error(f"Unsupported data type: {data_type}. Use 'file', 'url', 'gcp', or 'aws' ")
            return []

    except Exception as e:
        logging.error(f"Error loading and preprocessing the data {e}")
        return []


def search_internet(query: str, num_results: int = 5) -> List[str]:
    """
    Searches the internet using the Bing API (can be extended to use other APIs)
    Args:
        query: The search query.
        num_results: The number of search results to return
    Returns:
        A list of web pages extracted from the search results.
    """
    try:
        search_url = f'https://www.bing.com/search?q={query}'
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        for item in soup.find_all('li', class_='b_algo'):
            link = item.find('a')['href']
            try:
                result = fetch_data_from_url(link)
                if result:
                     results.append(result)
            except Exception as e:
                logging.warning(f"Could not process {link} {e}")

            if len(results) >= num_results:
                break
        logging.info(f"Successfully searched for {query} and got {len(results)} results")
        return results
    except Exception as e:
        logging.error(f"Error searching the internet {e}")
        return []

def add_knowledge_graph_data(pairs: List[Tuple[str, str]], graph: any= None) -> List[Tuple[str,str]]:
    """
    Add data to the knowledge graph and use the graph to make additional information
    Args:
        pairs: List of pairs of the data
        graph: knowledge graph data type, usually networkx or rdflib
    Returns:
        List of pairs of the enhanced data
    """
    if graph is None:
        logging.info("No knowledge graph is available. No graph based enhancement will be performed")
        return pairs
    try:
        enhanced_pairs = []
        if isinstance(graph, nx.Graph):
            for question, answer in pairs:
                tokens = preprocess_text(question).split()
                for token in tokens:
                    if graph.has_node(token):
                         neighbors = list(graph.neighbors(token))
                         if neighbors:
                             neighbor_text = " ".join(neighbors)
                             enhanced_pairs.append((question, f"{answer} - {neighbor_text}"))
        elif isinstance(graph, rdflib.Graph):
            for question, answer in pairs:
               tokens = preprocess_text(question).split()
               for token in tokens:
                   query = f"""
                       SELECT ?o
                       WHERE {{
                         <{token}> ?p ?o
                       }}
                   """
                   res = graph.query(query)
                   for row in res:
                       enhanced_pairs.append((question, f"{answer} - {str(row[0])}"))
        logging.info(f"Successfully enhanced pairs using knowledge graph {len(enhanced_pairs)} added")
        return pairs + enhanced_pairs
    except Exception as e:
        logging.error(f"Failed to add knowledge graph data to pairs {e}")
        return pairs


def extract_and_process_text_from_web(urls: List[str]) -> List[Tuple[str,str]]:
    """
    Extract text from a list of urls and process the text, and return it as question answer pair
    Args:
       urls : a list of web urls
    Returns:
         a list of question-answer pairs with the format ("from url", text)
    """
    pairs = []
    for url in urls:
       text = fetch_data_from_url(url)
       if text:
          pairs.append(("from url", text))
    logging.info(f"Extracted and processed text from {len(urls)} urls")
    return pairs


# --- Vocabulary Class ---

class Vocabulary:
    """
    Creates a vocabulary object that maps tokens to unique numerical ids
    """
    def __init__(self, min_freq=MIN_FREQ, add_unk=True, max_size=MAX_VOCAB_SIZE):
        self.min_freq = min_freq
        self.add_unk = add_unk
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.idx = 0

        if add_unk:
            self.add_token('<unk>')
            self.unk_idx = self.word2idx['<unk>']
        else:
            self.unk_idx = None
        self.add_token('<pad>')
        self.pad_idx = self.word2idx['<pad>']
        self.add_token('<sos>')
        self.sos_idx = self.word2idx['<sos>']
        self.add_token('<eos>')
        self.eos_idx = self.word2idx['<eos>']

    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.lower().split()
        return tokens

    def build_vocab(self, text_list):
        for text in text_list:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

        # Sort words by frequency and take the most common ones
        common_words = self.word_counts.most_common(self.max_size - 4)  # Reserve space for special tokens
        for token, count in self.word_counts.items():
            if count >= self.min_freq:
                self.add_token(token)
    def add_token(self, token: str):
        """
        Add tokens to the vocab
        Args:
             token: the token to be added
        """
        if token not in self.word2idx:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def numericalize(self, text: str, add_sos_eos: bool = False) -> List[int]:
        """
        Convert text to ids
        Args:
             text : input text to convert
             add_sos_eos : add sos and eos tokens
        Returns:
            List of ids corresponding to the tokens
        """
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        if add_sos_eos:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert the ids to texts
        Args:
             ids : list of ids
        Returns:
            Text correspongind to the ids
        """
        tokens = [self.idx2word.get(id, '<unk>') for id in ids]
        return ' '.join(tokens)

    def __len__(self) -> int:
        """
        Get the size of the vocab
        Returns:
             The length of the vocab
        """
        return len(self.word2idx)

# --- Dataset Classes ---

class ChatDataset(Dataset):
    """
    A custom dataset object that manages source and target text data
    """
    def __init__(self, pairs: List[Tuple[str, str]], vocab: Vocabulary,max_len = MAX_SEQ_LEN):
        """
        Initialize the dataset
        Args:
             pairs: list of pairs for the source and the target
             vocab: vocabulary object
        """
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Return the number of the items in the dataset
        Returns:
            Length of the pairs
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index idx
        Args:
            idx: Index of the data
        Returns:
            Tuple of tensors for the source and the target
        """
        src, tgt = self.pairs[idx]
        src_ids = self.vocab.numericalize(src)
        tgt_ids = self.vocab.numericalize(tgt, add_sos_eos=True)
        
        # Truncate sequences that are too long
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)
        
    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate the data batch using padding
        Args:
           batch: List of source and target tensors
        Returns:
           Padded tensor for the source and the target
        """
        src_tensors, tgt_tensors = zip(*batch)
        src_pad_seq = pad_sequence(src_tensors, batch_first=True, padding_value=self.vocab.pad_idx)
        tgt_pad_seq = pad_sequence(tgt_tensors, batch_first=True, padding_value=self.vocab.pad_idx)
        return src_pad_seq, tgt_pad_seq    
           
class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional info to the embeddings
    """
    def __init__(self, d_model,embedding_dim: int, dropout=DROPOUT, max_len=MAX_SEQ_LEN):
        """
        Initialize the positional encoding
        Args:
             embedding_dim: embedding dimension of the tokens
             dropout: dropout ratio for regularization
             max_len Maximum length of the sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding
        Args:
            x: Input tensor
        Returns:
            Output tensor with position encoding added.
        """
        x = x + self.pe[:x.size(0)].transpose(0, 1).to(x.device)
        return self.dropout(x)    
         
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout: float = DROPOUT, pad_idx: int = 0, device: any = DEVICE, sos_idx: int = 1):
        super(TransformerModel, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.transformer = Transformer(embedding_dim, num_heads, num_layers, hidden_dim, dropout, batch_first=True)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.pad_idx = pad_idx
        self.embedding_dim = embedding_dim
        self.sos_idx = sos_idx
        self.to(device)
    
    def generate_mask(self, src, pad_idx):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_mask = self.generate_mask(src, self.pad_idx)
        tgt_mask = self.generate_mask(tgt, self.pad_idx)
        tgt_seq_len = tgt.size(1)
        tgt_key_padding_mask = (tgt[:, :-1] == self.pad_idx).to(self.device)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device), diagonal=1).bool()
        tgt_mask = tgt_mask.logical_or(tgt_key_padding_mask.unsqueeze(1))
        src_embedded = self.embedding(src) * np.sqrt(self.embedding_dim)
        src_pos_encoded = self.pos_encoder(src_embedded)
        tgt_embedded = self.embedding(tgt) * np.sqrt(self.embedding_dim)
        tgt_pos_encoded = self.pos_encoder(tgt_embedded)

        output = self.transformer(
            src_pos_encoded,
            tgt_pos_encoded,
            src_key_padding_mask=~src_mask.squeeze(1).squeeze(1),
            tgt_key_padding_mask=~tgt_mask.squeeze(1).squeeze(1)
        )
        output = self.fc(output)
        return output

    def beam_search_decode(self, src_seq, vocab, beam_width=5, max_len=MAX_SEQ_LEN):
        self.eval()
        with torch.no_grad():
            src_tensor = torch.tensor(src_seq).unsqueeze(0).to(self.device)

            # Initialize beam with start token
            beam = [(torch.tensor([vocab.sos_idx]).to(self.device), 0.0)]

            for _ in range(max_len):
                new_beams = []
                for seq, log_prob in beam:
                    tgt_tensor = seq.unsqueeze(0).to(self.device)
                    output = self(src_tensor, tgt_tensor)
                    output = output[:, -1, :]  # Get prediction for the last token
                    log_probs = F.log_softmax(output, dim=-1)

                    top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, top_indices[0, i].unsqueeze(0)], dim=0)
                        new_log_prob = log_prob + top_log_probs[0, i].item()
                        new_beams.append((new_seq, new_log_prob))

                beam = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]  # Keep the top beams

                if all(seq[-1] == vocab.eos_idx for seq, _ in beam):
                    break

            best_seq, _ = beam[0]  # Get the sequence with highest log probability
            decoded_sequence = vocab.decode(best_seq.cpu().numpy().tolist())
            return decoded_sequence.replace('<sos>', '').replace('<eos>', '').strip()
    
    def generate(self, src: torch.Tensor, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        """
        Generate text
        Args:
            src: source text tensor
            max_len: maximum length of the generated text
        Returns:
            The generated tensor
        """
        self.eval()
        src = src.to(self.device)
        memory = self.embedding(src) * np.sqrt(self.embedding_dim)
        memory = self.pos_encoder(memory)

        src_mask = self.generate_mask(src, self.pad_idx)

        tgt_ids = torch.ones(1,1, device=self.device).fill_(self.sos_idx).long()
        
        for _ in range(max_len):
            
            tgt_embedded = self.embedding(tgt_ids) * np.sqrt(self.embedding_dim)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            tgt_mask = self.generate_mask(tgt_ids, self.pad_idx)
            
            output = self.transformer(
                memory,
                tgt_embedded,
                src_key_padding_mask=~src_mask.squeeze(1).squeeze(1),
                tgt_key_padding_mask=~tgt_mask.squeeze(1).squeeze(1)
            )
            output = self.fc(output[:, -1, :]) # Get the prediction for the last token
            
            output = torch.argmax(output, dim=-1)
            tgt_ids = torch.cat((tgt_ids, output.unsqueeze(0)), dim = 1)
            if output.item() == self.pad_idx:
                break
        
        return tgt_ids
    
def load_data(filename: str) -> List[Tuple[str, str]]:
    """Loads data from a JSON file, handling potential errors."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
                logging.error("Invalid JSON format: expected a list of pairs.")
                return []

        return [ (pair['question'], pair['answer']) for pair in data if isinstance(pair, dict) and 'question' in pair and 'answer' in pair ]

    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {filename}")
        return []
    except Exception as e:
         logging.error(f"Error loading data from {filename}: {e}")
         return []

def prepare_data(pairs: List[Tuple[str, str]], vocab: Vocabulary, test_size = 0.15) -> Tuple[DataLoader, DataLoader]:
        """Splits data and creates data loaders"""
        src_texts, tgt_texts = zip(*pairs)
        train_pairs, val_pairs  = train_test_split(list(zip(src_texts, tgt_texts)), test_size=test_size)
        train_dataset = ChatDataset(train_pairs, vocab)
        val_dataset = ChatDataset(val_pairs, vocab)

        train_loader = DataLoader(train_dataset, batch_size=LARGE_BATCH_SIZE , shuffle=True, collate_fn=train_dataset.collate_fn, num_workers = NUM_WORKERS, pin_memory=True) #Use multiple CPU Cores
        val_loader = DataLoader(val_dataset, batch_size=LARGE_BATCH_SIZE , shuffle=False, collate_fn=val_dataset.collate_fn, num_workers = NUM_WORKERS, pin_memory = True)
        return train_loader, val_loader

# --- Internet Search and Content Retrieval ---
def search_internet(query: str, num_results: int = 3) -> List[str]:
    """Searches the internet and returns snippets from relevant articles."""
    try:
        search_results = search(query, num_results = num_results, stop = num_results, pause = 2)
        content_list = []
        for url in search_results:
           article = newspaper.Article(url)
           article.download()
           article.parse()
           content_list.append(article.text)
        return content_list

    except Exception as e:
          logging.error(f"Error searching the internet for query: {query}, Error: {e}")
          return []

def scrape_website(url: str) -> str:
    """Scrapes text from a website using BeautifulSoup."""
    try:
        response = requests.get(url, timeout = 10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except Exception as e:
         logging.error(f"Error scraping website from {url}: {e}")
         return ""


def retrieve_data(query:str, source = 'internet') -> List[str]:
    """Retrieves content from specified source. """
    if source == 'internet':
       return search_internet(query)
    elif source == 'website':
          return [scrape_website(query)]
    else:
          logging.warning(f"Invalid source: {source}, No content will be retrieved")
          return []


def process_retrieved_content(content_list: List[str]) -> str:
      """Processes retrieved content and returns a concatenated string."""
      processed_content_list = [advanced_text_preprocessing(text) for text in content_list]
      return " ".join(processed_content_list)

      
def evaluate(model: nn.Module,val_loader: DataLoader, criterion: nn.CrossEntropyLoss, vocab, writer, step: int ,data_loader: DataLoader) -> Tuple[float, float, float]:
    """
    Evaluate model performance
    Args:
       model: the model to evaluate
       data_loader: the data loader for the data
       criterion : the loss function
    Returns:
        Tuple of eval loss, perplexity, and accuracy
    """
    model.eval()
    epoch_loss = 0
    num_batches = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            output = model(src_batch, tgt_batch[:, :-1])
            output = output.view(-1, output.size(-1))
            tgt_batch_reshape = tgt_batch[:,1:].reshape(-1)
            loss = criterion(output, tgt_batch_reshape)
            val_loss += loss.item()
            num_batches += 1

            epoch_loss += loss.item()
            num_batches += 1


            # Accuracy Calculation
            pred_tokens = torch.argmax(output, dim=-1)
            correct_tokens += (pred_tokens == tgt_batch_reshape).sum().item()
            total_tokens += tgt_batch_reshape.size(0)

    avg_val_loss = val_loss / num_batches
    if writer:
        writer.add_scalar("validation_loss", avg_val_loss, step)
    model.train() 
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    perplexity = np.exp(log_loss)

    return log_loss, perplexity, accuracy
def respond(model, query: str, vocab: Vocabulary, search_enabled: bool = True) -> str:
    model.eval()
    processed_query = advanced_text_preprocessing(query, lemmatize=True)
    query_ids = vocab.numericalize(processed_query)

    if len(query_ids) < 4:
        if search_enabled:
            retrieved_content = retrieve_data(query)
            if retrieved_content:
                processed_content = process_retrieved_content(retrieved_content)
                combined_query = f"{query} {processed_content}"
                combined_query_ids = vocab.numericalize(combined_query)
                response = beam_search_decode(model, combined_query_ids, vocab, max_len=1000)
                return response or get_default_response()
            return get_default_response()
        return get_default_response("Please provide more context to the query.")
    
    response = beam_search_decode(model, query_ids, vocab, max_len=1000)
    return response or get_default_response()


def get_default_response(message: str = "I'm not sure how to respond to this. Can you ask something else?") -> str:
    return message

def respond(model, query: str, vocab: Vocabulary, search_enabled = True) -> str:
    """Generates a response to user input."""
    model.eval()

    processed_query = advanced_text_preprocessing(query, lemmatize = True) #Preprocess the User Query
    query_ids = vocab.numericalize(processed_query)


    # Check if the query is very short
    if len(query_ids) < 4:
        if search_enabled:
           retrieved_content = retrieve_data(query)  # Search content if query is short and search is enabled
           if retrieved_content:
                 processed_content = process_retrieved_content(retrieved_content)
                 combined_query = query + " " + processed_content #Combine original query with retrieved content
                 combined_query_ids = vocab.numericalize(combined_query)
                 response = beam_search_decode(model, combined_query_ids, vocab, max_len = 1000)
                 return response
           else:
               return "I'm not sure how to respond to this.  Can you ask something else?"

        else:
             return "Please provide more context to the query"  #If search is disabled
    else:
        response = beam_search_decode(model, query_ids, vocab, max_len = 100) #Generate using Beam Search
        if response:
            return response

        else:
           return "I'm not sure how to respond to this.  Can you ask something else?"
def train_epoch(
    model: nn.Module, 
    train_loader: DataLoader,
    val_loader: DataLoader, 
    scheduler: optim.lr_scheduler.ReduceLROnPlateau, 
    vocab, 
    optimizer: optim.Optimizer, 
    criterion: Any, 
    clip: float = MAX_GRAD_NORM,
    epochs: int = EPOCH, 
    patience: int = PATIENCE, 
    eval_frequency: int = 1, 
    writer=None               
) -> Tuple[float, float, float]:
    
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index = vocab.pad_idx) # Loss Calculation with ignoring padding tokens
    best_val_loss = float('inf') # Track validation loss for early stopping
    patience_count = 0
    best_model_path = "best_model.pth" # Path where the best model will be saved
    step = 0  #Global training step counter
    
    for epoch in range(epochs): # Loop through epochs
        model.train()
        epoch_loss = 0
        num_batches = 0
        total_tokens = 0
        correct_tokens = 0
        start_time = time.time()
        
        for src_batch, tgt_batch in train_loader:
            step += 1
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch[:, :-1])

            output = output.view(-1, output.size(-1))
            tgt_batch_reshape = tgt_batch[:,1:].reshape(-1)
            loss = criterion(output, tgt_batch_reshape)
            
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Accuracy Calculation
            pred_tokens = torch.argmax(output, dim=-1)
            correct_tokens += (pred_tokens == tgt_batch_reshape).sum().item()
            total_tokens += tgt_batch_reshape.size(0)
        
            if step % 10 == 0 and writer:
                writer.add_scalar("training_loss", loss.item(), step)
        
        avg_loss = epoch_loss / num_batches
        accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss)


        #Validation
        val_loss = float('inf') # Initialize val_loss before conditional evaluation
        if (epoch + 1) % eval_frequency == 0:
            val_loss = evaluate(model, val_loader, criterion, vocab, writer, step) #Evaluate the model on validation set
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0 # Reset patience if validation loss improves
                torch.save(model.state_dict(), best_model_path)
                logging.info(f'New Best Validation Loss: {best_val_loss:.4f} Model Saved! ')
            else:
                patience_count += 1
            
            scheduler.step(val_loss)  #Adjust learning rate based on validation loss

        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Time:{elapsed_time:.2f} seconds, Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}')

        if patience_count >= patience:
            logging.info('Early stopping triggered! ')
            break
    
    if writer:
        writer.close()
    logging.info(f'Training is completed, Best model saved at {best_model_path}')

    return avg_loss, perplexity, accuracy
def evaluate(model, val_loader, criterion, vocab, writer, step):
    """Evaluates model performance on validation data"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
              src_batch = src_batch.to(DEVICE)
              tgt_batch = tgt_batch.to(DEVICE)
              output = model(src_batch, tgt_batch[:, :-1])
              output_reshape = output.reshape(-1, output.shape[-1])
              tgt_batch_reshape = tgt_batch[:,1:].reshape(-1)
              loss = criterion(output_reshape, tgt_batch_reshape)
              val_loss += loss.item()

    val_loss /= len(val_loader)
    if writer:
        writer.add_scalar("validation_loss", val_loss, step)
    return val_loss

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer,
                      criterion: any, scheduler: any = None, num_epochs: int = NUM_EPOCHS,
                       eval_steps: int = EVAL_STEPS, save_steps: int = SAVE_STEPS, model_path: str = "transformer_model.pt"):
    """
    Training and evaluation routine
    Args:
        model: model to train
        train_loader: data loader for the training set
        val_loader: data loader for the validation set
        optimizer: the optimizer to use
        criterion: the loss function
        scheduler: learning rate scheduler
        num_epochs : number of training epochs
        eval_steps: the interval to evaluate the model
        save_steps: the interval to save the model
        model_path: the path to save the model
    """
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        train_loss, train_perplexity, train_acc = train_epoch(model, train_loader, optimizer, criterion, clip=MAX_GRAD_NORM)
        end_epoch_time = time.time()

        logging.info(f"Epoch: {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Perplexity: {train_perplexity:.4f} | Train Accuracy: {train_acc:.4f}")
        logging.info(f"Epoch Time: {(end_epoch_time - start_epoch_time):.2f}s")

        if scheduler:
          scheduler.step(train_loss)

        if (epoch + 1) % eval_steps == 0:
            val_loss, val_perplexity, val_acc = evaluate(model, val_loader, criterion)
            logging.info(f"Evaluation: | Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f} | Val Accuracy: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"Saving model with validation loss of {best_val_loss} at epoch {epoch + 1}")
                torch.save(model.state_dict(), model_path)

        if (epoch+1) % save_steps == 0:
           logging.info(f"Saving model at step {epoch + 1}")
           torch.save(model.state_dict(), f"model_at_{epoch+1}.pt")

    end_time = time.time()
    logging.info(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
def adjust_batch_size(current_batch_size: int, memory_usage: int, target_memory_percentage: float = 0.85) -> int:
    """
    Dynamically adjust the batch size based on memory usage
    Args:
       current_batch_size: current batch size being used
       memory_usage: the memory usage in percentage
       target_memory_percentage: the target memory percentage
    Returns:
        Adjusted batch size
    """
    if memory_usage >= target_memory_percentage:
        new_batch_size = max(current_batch_size // 2, 1)
        logging.warning(f"Memory usage at {memory_usage:.2f}%, reducing batch size to {new_batch_size}")
        return new_batch_size
    elif memory_usage <= (target_memory_percentage/2) and current_batch_size < LARGE_BATCH_SIZE:
        new_batch_size = min(current_batch_size * 2, LARGE_BATCH_SIZE)
        logging.info(f"Memory usage low at {memory_usage:.2f}%, increasing batch size to {new_batch_size}")
        return new_batch_size
    return current_batch_size
def monitor_resource_usage(batch_size_queue: queue.Queue, stop_event: threading.Event, interval: float = 10, initial_batch_size: int = 32):
    """
    Monitors resource usage (memory) and adjusts batch size accordingly.
    This function runs in a separate thread.

    Args:
        batch_size_queue: A queue to communicate the adjusted batch size to other parts of the program.
        stop_event: A threading.Event to signal when the monitoring thread should stop.
        interval: The time interval (in seconds) between memory usage checks.
        initial_batch_size: The initial batch size to start with.
    """
    current_batch_size = initial_batch_size
    while not stop_event.is_set():
        try:
            memory_usage = psutil.virtual_memory().percent / 100.0  # Get memory usage as a float (0.0 to 1.0)
            current_batch_size = adjust_batch_size(current_batch_size, memory_usage)
            batch_size_queue.put(current_batch_size)
            time.sleep(interval)
        except Exception as e:
            logging.error(f"Error in resource monitoring thread: {e}")
            break # Exit loop if an unexpected error happens
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    batch_size_queue = queue.Queue()
    stop_event = threading.Event()
    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resource_usage, args=(batch_size_queue, stop_event), daemon=True)
    monitor_thread.start()

    #Load the data
    logging.info('Loading Data ....')
    pairs = load_data("qa_pairs.json")
    if not pairs:
         logging.error('No training data was loaded')
         exit(1)

    # Build Vocabulary
    logging.info('Building Vocabulary...')
    all_texts = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    vocab = Vocabulary()
    vocab.build_vocab(all_texts)
    logging.info(f"Vocabulary Size : {len(vocab)}")

    # Prepare DataLoaders
    logging.info('Preparing Data Loaders...')
    train_loader, val_loader = prepare_data(pairs, vocab)


    # Initialize Model
    logging.info('Initializing the Model...')
    model = TransformerModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx = vocab.pad_idx,
        device=DEVICE
    ).to(DEVICE)
    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")


    # Initialize Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience= 10, verbose=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Train the Model
    logging.info('Start Training...')
    train(model, train_loader, val_loader, optimizer, scheduler, vocab, writer=writer)

    # Interact with the chatbot
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = respond(model, user_input, vocab) #Search can be enabled / disabled
        print("Bot:", response)
