import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import threading
import time
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

class AdvancedAutoLearnerBot:
    def __init__(self):
        self.knowledge_base = {}
        self.learned_sources = set()
        self.file_path = "auto_knowledge.json"
        self.load_knowledge()
        self.search_engines = {
            'wikipedia': 'https://fa.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json',
            'stackoverflow': 'https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={}&site=stackoverflow',
            'technical_blogs': 'https://api.github.com/search/repositories?q={}+in:readme&sort=updated'
            , 'google': 'https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={}' # You need to setup this
        }
        self.api_keys = {
           'google_api_key': None , #Please add google api key
            'google_cse_id': None #Please add Google Custom search engine id
        }
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1, 2))
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        self.data_lock = threading.Lock()
        self.training_data = pd.DataFrame(columns=['query', 'content', 'label'])
        self.preprocessor = None # Initialize to None for now
        self.model = None 
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=100)
        self.executor = ThreadPoolExecutor(max_workers=10) #Thread pool for async tasks
        self.learning_rate = 0.02
        self.batch_size = 128
        self.num_epochs = 200
        self.num_layers = 30
        self.optimization_strategy = "adaptive_gradient" # "adam", "sgd", "momentum" , adaptive_gradient
        self.learning_history = []
        self.model_initialized = False
        self.thread_lock = threading.Lock()
        self.max_knowledge_size = 2000
        self.knowledge_queue = []
        self.knowledge_access_counts = {}
        self.max_retries = 3 # Max number of retries for network operations
        self.retry_delay = 2  # Delay between retries in seconds
        self.learning_threshold = 0.8 # Learning threshold for adding knowledge
        self.global_context = {} # Global context for tracking previous interactions
        self.interaction_limit = 50 #Number of interactions to keep track of in global context
        self.message_history = {}
        self.use_gpu_accelaration = False #Flag for use gpu accelaration
        self.use_tpu_accelaration = False #Flag for use tpu accelaration
        self.training_history = []
        self.training_process_lock = threading.Lock()
        self.use_external_apis = True # Flag for using external api's
        self.feature_selection = "tfidf" # "tfidf", "pca", or "all"
        self.cross_validation_folds = 5 # Number of folds for cross-validation
        self.model_version = 1.0 #Model versioning for future updates
        self.response_generation_strategy = "knn" # "knn", "hybrid_nn"
        self.context_window_size = 5
        self.context_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 1)) 
        self.context_model = None
        self.context_data = [] 
        self.context_labels = [] 
        self.knowledge_retention_ratio = 0.95 # Retention ratio for removing old data
        self.knowledge_base_limit = 10000 # Maximum knowledge base size

    def load_knowledge(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            print("Knowledge base file not found, creating a new one.")
            self.knowledge_base = {}
        except json.JSONDecodeError:
            print("Error decoding JSON from knowledge base. Clearing and starting fresh.")
            self.knowledge_base = {}
            
        for key in list(self.knowledge_base.keys()):
            if not isinstance(self.knowledge_base[key], dict):
               print(f"Incorrect formatting found for key {key}, removing")
               del self.knowledge_base[key]

    def save_knowledge(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
             json.dump(self.knowledge_base, f, ensure_ascii=False, indent=4)

    def online_learn(self, query, label=None):
        """
        Learns from online resources and adds to the knowledge base.
        Uses multithreading to avoid blocking and for asynchronous tasks.
        """
        if not self.use_external_apis:
            print("External API usage is disabled, skipping online learning.")
            return
        
        if query in self.knowledge_base and isinstance(self.knowledge_base.get(query), dict):
            print(f"'{query}' already in knowledge base.")
            return

        print(f"🔎 Learning '{query}' from the web...")

        # Create a unique identifier for this query
        query_id = hashlib.sha256(query.encode('utf-8')).hexdigest()

        if query_id in self.knowledge_base and isinstance(self.knowledge_base[query_id], dict):
           print(f"'{query}' already in knowledge base.")
           return

        # Define a function to execute the learning in a separate thread
        def threaded_learn():
            learned_data = []
            for source, url in self.search_engines.items():
                try:
                     if source == "google" and self.api_keys["google_api_key"] is not None and self.api_keys["google_cse_id"] is not None:
                       url = url.format(google_api_key=self.api_keys["google_api_key"],google_cse_id=self.api_keys["google_cse_id"], quote_plus(query))
                     else:
                       url = url.format(quote_plus(query))
                     
                     response = self._make_http_request(url)
                     data = response.json()
                     learned_data += self._process_source(source, data)
                except Exception as e:
                    print(f"Error in source {source}: {str(e)}")

            if learned_data:
                self._extract_key_info(query, learned_data, label)
                self.save_knowledge()

        # Run the learning in a separate thread
        self.executor.submit(threaded_learn)

    def _make_http_request(self, url, method='get', data=None, headers=None):
        retries = 0
        while retries <= self.max_retries:
            try:
                if method.lower() == 'get':
                    response = requests.get(url, headers=headers)
                elif method.lower() == 'post':
                     response = requests.post(url, json=data, headers=headers)
                else:
                     raise ValueError("Invalid HTTP method")

                response.raise_for_status() # Raises an exception for non-200 status codes
                return response
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"Request failed: {e} - Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        raise Exception(f"Failed to fetch {url} after {self.max_retries} retries.")

    def _process_source(self, source, data):
        """Processes data from different sources."""
        processed = []

        if source == 'wikipedia':
            for item in data.get('query', {}).get('search', [])[:3]:
                page_url = f"https://fa.wikipedia.org/?curid={item['pageid']}"
                content = self._scrape_page(page_url)
                if content:
                    processed.append(content)

        elif source == 'stackoverflow':
            for item in data.get('items', [])[:3]:
                content = self._clean_html(item['body_markdown'])
                if content:
                    processed.append(f"{item['title']}\n{content}")
        elif source == 'technical_blogs':
              for item in data.get('items', [])[:3]:
                    repo_url = item.get('html_url')
                    if repo_url:
                       content = self._scrape_page(repo_url)
                       if content:
                         processed.append(f"{item.get('name')}\n{content}")
        elif source == 'google':
               for item in data.get('items', [])[:3]:
                page_url = item.get('link')
                if page_url:
                  content = self._scrape_page(page_url)
                  if content:
                    processed.append(f"{item.get('title')}\n{content}")
        return processed

    def _scrape_page(self, url):
        try:
            response = self._make_http_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = ' '.join(p.get_text() for p in soup.find_all('p'))
            return self._clean_text(text_content)
        except Exception as e:
            print(f"Error scraping page {url}: {e}")
            return None

    def _extract_key_info(self, query, learned_data, label=None):
        """Extracts key information and adds to knowledge base."""
        cleaned_data = [data for data in learned_data if data is not None and data.strip()]

        if cleaned_data:
          combined_content = ' '.join(cleaned_data)
          query_id = hashlib.sha256(query.encode('utf-8')).hexdigest()

          # Add the new data to the knowledge base
          with self.data_lock:
                if query_id not in self.knowledge_base or not isinstance(self.knowledge_base.get(query_id), dict):
                   self.knowledge_base[query_id] = {
                      "query": query,
                      "content": combined_content,
                      "label": label,
                      "added_timestamp": time.time(),
                      "access_count": 0
                   }

                   self.knowledge_queue.append(query_id)
                   self.knowledge_access_counts[query_id] = 0 
                   self._manage_knowledge_size() 
                   self._update_training_data(query, combined_content, label)
          print(f"✔️ Learned about '{query}'.")
        else:
            print(f"❌ No significant data found for '{query}'.")

    def _clean_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.stripped_strings)

    def _clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^آ-یa-zA-Z0-9\s.,?!]+', '', text) #Keep only letters, numbers, spaces, and punctuation marks
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _update_training_data(self, query, content, label):
         """Updates the internal training dataset."""
         with self.thread_lock:
            new_data = pd.DataFrame([{'query': query, 'content': content, 'label': label}])
            self.training_data = pd.concat([self.training_data, new_data], ignore_index=True)
            self.training_data = self.training_data.drop_duplicates(subset=['query'],keep='last')
            self._train_model()

    def _train_model(self):
        """Trains the machine learning model."""
        if self.training_data.empty:
           print("No training data available yet.")
           return

        print("🏋️‍♀️ Training the model...")

        with self.training_process_lock:
            if self.feature_selection == "tfidf" or self.feature_selection == "all":
              X = self.tfidf_vectorizer.fit_transform(self.training_data['content']).toarray()
            elif self.feature_selection == "pca":
              X = self.tfidf_vectorizer.fit_transform(self.training_data['content']).toarray()
              X = self.pca.fit_transform(X)
            
            y = self.training_data['label'].astype('category').cat.codes if self.training_data['label'] is not None else np.zeros(len(self.training_data))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if not self.model_initialized:
              self._initialize_model()
              self.model_initialized = True
            
            if self.use_gpu_accelaration:
               print("Using GPU acceleration for training.")
            elif self.use_tpu_accelaration:
              print("Using TPU acceleration for training.")

            self._train_model_with_optimization(X_train, y_train, X_test, y_test)

            if self.response_generation_strategy == "knn":
             self.knn_classifier.fit(X_train, y_train)

    def _initialize_model(self):
      
       print("Model initialization...")
       if self.optimization_strategy == "adaptive_gradient":
            self.model = self.AdaptiveGradientModel(input_size=self.tfidf_vectorizer.max_features, hidden_layers=[(256, 'relu'), (128, 'relu'), (64,'relu')], output_size=len(set(self.training_data['label'])) if self.training_data['label'] is not None and  self.training_data['label'].nunique() > 0 else 1, learning_rate=self.learning_rate, batch_size=self.batch_size)
       
    def _train_model_with_optimization(self, X_train, y_train, X_test, y_test):

      if not hasattr(self,"model") or self.model is None:
            print("No model initialized")
            return
      
      print("Training...")
      losses = []
      accuracies = []
      start_time = time.time()
      for epoch in range(self.num_epochs):
          epoch_loss = 0
          epoch_acc = 0
          for i in range(0, X_train.shape[0], self.batch_size):
             x_batch = X_train[i:i+self.batch_size]
             y_batch = y_train[i:i+self.batch_size]
             loss, accuracy = self.model.train_step(x_batch, y_batch)
             epoch_loss += loss
             epoch_acc += accuracy
          epoch_loss /= (X_train.shape[0]/self.batch_size)
          epoch_acc /= (X_train.shape[0]/self.batch_size)
          test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
          losses.append(epoch_loss)
          accuracies.append(epoch_acc)
          end_time = time.time()
          time_taken = end_time-start_time

          print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}, time: {time_taken:.2f}")
          start_time = time.time()

      training_info = {
        "final_loss": losses[-1] if losses else None,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "losses": losses,
        "accuracies": accuracies,
        "model_version": self.model_version,
        "timestamp": time.time()
      }
      self.training_history.append(training_info)
      print("Training complete")

    def query(self, query_text):
        
        """Queries the knowledge base and returns the best response using a chosen strategy."""
        query_text = self._clean_text(query_text)

        if query_text in self.global_context:
             query_context = self.global_context[query_text]
             query_context.append(query_text)
             self.global_context[query_text] = query_context[-self.context_window_size:]
             print("Using conversation history for response.")
             response = self._respond_with_context(query_text)
             return response
           
        print(f"❓ Query: {query_text}")

        if self.response_generation_strategy == "knn":
            response = self._respond_with_knn(query_text)
        elif self.response_generation_strategy == "hybrid_nn":
           response = self._respond_with_hybrid_nn(query_text)
        
        
        #Update access counts and conversation history
        self._update_knowledge_access(query_text)
        self._update_global_context(query_text)
        return response
    
    def _update_knowledge_access(self, query):
         query_id = hashlib.sha256(query.encode('utf-8')).hexdigest()

         with self.data_lock:
              if query_id in self.knowledge_base:
                  self.knowledge_base[query_id]["access_count"] += 1
                  self.knowledge_access_counts[query_id] +=1
                  
    def _update_global_context(self, query):
      if query in self.global_context:
          self.global_context[query].append(query)
          self.global_context[query] = self.global_context[query][-self.context_window_size:]
      else:
           self.global_context[query] = [query]
    
    def _respond_with_knn(self, query_text):
         """Responds using the KNN classification and TF-IDF vectorization."""

         if not hasattr(self,"knn_classifier") or self.knn_classifier is None:
            print("KNN Model is not trained yet.")
            return "Sorry, I don't have enough information to respond yet. Please try again later."

         try:
           query_vector = self.tfidf_vectorizer.transform([query_text]).toarray()
           predicted_label = self.knn_classifier.predict(query_vector)[0]
           
           # Retrieve the corresponding label from the data
           if hasattr(self.training_data, 'label') and len(set(self.training_data['label'])) >0:
            label_categories = self.training_data['label'].astype('category')
            original_label = label_categories.cat.categories[predicted_label]
            
            #Search the original knowledge base
            matched_entries = [entry for entry in self.knowledge_base.values() if entry.get('label') == original_label]
            if matched_entries:
                best_match = max(matched_entries, key=lambda entry: self._similarity_score(query_text, entry['content']))
                return best_match.get('content') if best_match.get('content') else "No relevant information found."
            else:
                 return "No information found with this label"
           else:
              return "No information found, model training was not finished properly"
           
         except Exception as e:
            print(f"Error in KNN response: {e}")
            return "Sorry, I encountered an error while trying to respond."
    
    def _similarity_score(self, query_text, content):
            """Calculates a similarity score between query and content."""
            
            query_vector = self.tfidf_vectorizer.transform([query_text]).toarray()
            content_vector = self.tfidf_vectorizer.transform([content]).toarray()
            
            try:
             score = np.dot(query_vector, content_vector.T)[0][0]
             return score
            except Exception as e:
             print(f"Error in similirity check: {e}")
             return 0.0

    def _respond_with_hybrid_nn(self, query_text):
            """Responds using the neural network model."""

            if not hasattr(self, "model") or self.model is None:
                return "Sorry, the neural network model is not initialized."

            try:
                 query_vector = self.tfidf_vectorizer.transform([query_text]).toarray()

                 # Use trained model to get the predicted distribution
                 prediction_distribution = self.model.predict(query_vector)
                 predicted_label = np.argmax(prediction_distribution, axis=1)[0]

                 # Retrieve the corresponding label from the data
                 if hasattr(self.training_data, 'label') and len(set(self.training_data['label'])) > 0:
                    label_categories = self.training_data['label'].astype('category')
                    original_label = label_categories.cat.categories[predicted_label]

                     # Search the original knowledge base
                    matched_entries = [entry for entry in self.knowledge_base.values() if entry.get('label') == original_label]
                    if matched_entries:
                         best_match = max(matched_entries, key=lambda entry: self._similarity_score(query_text, entry['content']))
                         return best_match.get('content') if best_match.get('content') else "No relevant information found."
                    else:
                         return "No information found with this label"
                 else:
                      return "No information found, model training was not finished properly"
            except Exception as e:
                print(f"Error in neural network response: {e}")
                return "Sorry, I encountered an error while trying to respond."

    def _respond_with_context(self, query_text):
       """Responds by leveraging the context from previous queries."""
       if self.context_model is None:
             self._train_context_model()
       
       context_vector = self.context_vectorizer.transform([' '.join(self.global_context[query_text])]).toarray()
       predicted_label = self.context_model.predict(context_vector)[0]

       # Retrieve the corresponding label from the data
       if hasattr(self.training_data, 'label') and len(set(self.training_data['label'])) > 0:
            label_categories = self.training_data['label'].astype('category')
            original_label = label_categories.cat.categories[predicted_label]

            # Search the original knowledge base
            matched_entries = [entry for entry in self.knowledge_base.values() if entry.get('label') == original_label]
            if matched_entries:
                best_match = max(matched_entries, key=lambda entry: self._similarity_score(query_text, entry['content']))
                return best_match.get('content') if best_match.get('content') else "No relevant information found."
            else:
                 return "No information found with this label"
       else:
              return "No information found, model training was not finished properly"

    def _train_context_model(self):
            """Trains the context model using conversation history."""
            print("Training context model")
            for key, value in self.global_context.items():
              self.context_data.append(' '.join(value))
              self.context_labels.append(key)
            
            if not self.context_data:
                return
            
            context_vectors = self.context_vectorizer.fit_transform(self.context_data).toarray()
            context_labels = pd.Series(self.context_labels).astype('category').cat.codes
            self.context_model = KNeighborsClassifier(n_neighbors=3, metric='cosine')
            self.context_model.fit(context_vectors, context_labels)
            print("Context model trained.")

    def _manage_knowledge_size(self):
        """Manages the size of the knowledge base by removing the least accessed items."""
        with self.data_lock:
              while len(self.knowledge_base) > self.knowledge_base_limit:
                  
                  if self.knowledge_queue:
                        least_accessed = min(self.knowledge_queue, key=lambda k : self.knowledge_access_counts.get(k,0))
                        if least_accessed and least_accessed in self.knowledge_base:
                            del self.knowledge_base[least_accessed]
                            del self.knowledge_access_counts[least_accessed]
                            self.knowledge_queue.remove(least_accessed)
                            print("Removing least accessed data")
                  else:
                    
                    keys_to_remove = list(self.knowledge_base.keys())[:int((1-self.knowledge_retention_ratio) * len(self.knowledge_base))]
                    for key in keys_to_remove:
                        del self.knowledge_base[key]
                        if key in self.knowledge_access_counts:
                           del self.knowledge_access_counts[key]
                        print("Removing old data")
                    self.knowledge_queue = []

    class AdaptiveGradientModel:
        def __init__(self, input_size, hidden_layers, output_size, learning_rate, batch_size, use_gpu=False, use_tpu=False):
            self.input_size = input_size
            self.hidden_layers = hidden_layers
            self.output_size = output_size
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.use_gpu = use_gpu
            self.use_tpu = use_tpu

            self.params = self._initialize_params()
            self.momentum = {}
            self.velocity = {}
            for layer_key in self.params:
                if layer_key.startswith('W'):
                  self.momentum[layer_key] = np.zeros_like(self.params[layer_key])
                  self.velocity[layer_key] = np.zeros_like(self.params[layer_key])

        def _initialize_params(self):
            params = {}
            sizes = [self.input_size] + [layer[0] for layer in self.hidden_layers] + [self.output_size]
            for i in range(len(sizes) - 1):
                params[f'W{i+1}'] = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                params[f'b{i+1}'] = np.zeros((1, sizes[i+1]))
            return params

        def _softmax(self, x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        def _relu(self, x):
            return np.maximum(0, x)

        def _relu_derivative(self, x):
            return (x > 0).astype(float)

        def _forward(self, X):
            activations = {'A0': X}
            for i, layer in enumerate(self.hidden_layers):
                W = self.params[f'W{i+1}']
                b = self.params[f'b{i+1}']
                Z = np.dot(activations[f'A{i}'], W) + b
                A = self._relu(Z)
                activations[f'Z{i+1}'] = Z
                activations[f'A{i+1}'] = A

            W_output = self.params[f'W{len(self.hidden_layers)+1}']
            b_output = self.params[f'b{len(self.hidden_layers)+1}']
            Z_output = np.dot(activations[f'A{len(self.hidden_layers)}'], W_output) + b_output
            A_output = self._softmax(Z_output)
            activations['Z_output'] = Z_output
            activations['A_output'] = A_output
            return activations

        def _backward(self, X, y, activations):
            m = y.shape[0]
            grads = {}
            dZ_output = activations['A_output'] - self._one_hot_encode(y)

            grads[f'dW{len(self.hidden_layers)+1}'] = np.dot(activations[f'A{len(self.hidden_layers)}'].T, dZ_output) / m
            grads[f'db{len(self.hidden_layers)+1}'] = np.sum(dZ_output, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ_output, self.params[f'W{len(self.hidden_layers)+1}'].T)

            for i in range(len(self.hidden_layers), 0, -1):
                dZ = dA_prev * self._relu_derivative(activations[f'Z{i}'])
                grads[f'dW{i}'] = np.dot(activations[f'A{i-1}'].T, dZ) / m
                grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
                dA_prev = np.dot(dZ, self.params[f'W{i}'].T)

            return grads

        def _one_hot_encode(self, y):
          num_classes = self.output_size
          one_hot = np.zeros((len(y), num_classes))
          one_hot[np.arange(len(y)), y] = 1
          return one_hot

        def _cross_entropy_loss(self, y_true, y_pred):
           m = y_true.shape[0]
           log_probs = -np.log(y_pred[np.arange(m), y_true] + 1e-8)
           return np.mean(log_probs)
        
        def _accuracy(self, y_true, y_pred):
              y_pred_labels = np.argmax(y_pred, axis=1)
              return np.mean(y_pred_labels == y_true)

        def _update_params_adaptive_gradient(self, grads, beta1=0.9, beta2=0.999, epsilon=1e-8):
          """
            Updates the model parameters using Adaptive Gradient method (Momentum + RMSprop)
          """
          for key in self.params:
              if key.startswith('W'):
                 self.momentum[key] = beta1*self.momentum[key] + (1-beta1)*grads[key]
                 self.velocity[key] = beta2*self.velocity[key] + (1-beta2)*(grads[key]**2)

                 momentum_corrected = self.momentum[key] / (1- beta1)
                 velocity_corrected = self.velocity[key] / (1-beta2)
                 self.params[key] -= self.learning_rate * momentum_corrected / (np.sqrt(velocity_corrected) + epsilon)

              elif key.startswith('b'):
                 self.params[key] -= self.learning_rate * grads[key]
        
        def train_step(self, X, y):
            activations = self._forward(X)
            grads = self._backward(X, y, activations)
            self._update_params_adaptive_gradient(grads)
            loss = self._cross_entropy_loss(y, activations['A_output'])
            accuracy = self._accuracy(y, activations['A_output'])
            return loss, accuracy
        
        def predict(self, X):
              activations = self._forward(X)
              return activations['A_output']
        
        def evaluate(self, X, y):
              activations = self._forward(X)
              loss = self._cross_entropy_loss(y, activations['A_output'])
              accuracy = self._accuracy(y, activations['A_output'])
              return loss, accuracy


if __name__ == "__main__":
    bot = AdvancedAutoLearnerBot()

    # Example usage
    while True:
        user_query = input("Enter your query: (type 'exit' to quit): ").strip()

        if user_query.lower() == 'exit':
            break

        # Handle learning commands
        if user_query.startswith("learn:"):
            query_parts = user_query[6:].split("label:")
            learn_query = query_parts[0].strip()
            label = query_parts[1].strip() if len(query_parts) > 1 else None
            bot.online_learn(learn_query, label)
            # Get response after learning
            response = bot.query(learn_query)
            print("\n🤖 Response:", response)
        else:
            # Handle regular queries
            response = bot.query(user_query)
            print("\n🤖 Response:", response)

        print("\n" + "-"*50 + "\n")

    bot.save_knowledge()
    print("Knowledge base saved. Goodbye!")