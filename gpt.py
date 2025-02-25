import json
import random
import re
import math
from collections import defaultdict, Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import ast
import inspect
import dill
import sys
import hashlib
from textwrap import dedent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SelfImprovingAI:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.error_log = []
        self.performance_history = []
        self.code_versions = {}
        self.current_version = "1.0.0"
        self.improvement_strategies = [
            self._optimize_hyperparameters,
            self._architectural_improvement,  
            self._feature_engineering
        ]
    def _augment_training_data(self):
        """تولید داده‌های مصنوعی برای بهبود آموزش"""
        synthetic_data = []
        for item in self.chatbot.memory:
            # ایجاد تغییرات ساده در جملات موجود
            augmented = re.sub(r'\b(hello|hi)\b', 'hey', item['user_input'], flags=re.IGNORECASE)
            if augmented != item['user_input']:
                synthetic_data.append({
                    'user_input': augmented,
                    'bot_response': item['bot_response']
                })
        # اضافه کردن به حافظه
        if synthetic_data:
            self.chatbot.memory.extend(synthetic_data)
            self.chatbot.train(self.chatbot.memory)
            return True
        return False
    def _save_snapshot(self, version):
        """ذخیره وضعیت فعلی چتبات"""
        snapshot = {
            'memory': dill.dumps(self.chatbot.memory),
            'models': {
                'knn': dill.dumps(self.chatbot.knn),
                'mlp': dill.dumps(self.chatbot.mlp),
                'logistic': dill.dumps(self.chatbot.logistic)
            },
            'vectorizer': dill.dumps(self.chatbot.vectorizer),
            'embeddings': dill.dumps(self.chatbot.embedding_matrix)
        }
        self.code_versions[version] = snapshot
    
    def _rollback(self, version):
        """بازگردانی به نسخه قبلی"""
        if version in self.code_versions:
            snapshot = self.code_versions[version]
            self.chatbot.memory = dill.loads(snapshot['memory'])
            self.chatbot.knn = dill.loads(snapshot['models']['knn'])
            self.chatbot.mlp = dill.loads(snapshot['models']['mlp'])
            self.chatbot.logistic = dill.loads(snapshot['models']['logistic'])
            self.chatbot.vectorizer = dill.loads(snapshot['vectorizer'])
            self.chatbot.embedding_matrix = dill.loads(snapshot['embeddings'])
            return True
        return False
    
    def _generate_code(self, problem_type):
        """تولید کد هوشمند برای حل مشکل"""
        code_templates = {
            'hyperparameters': """
                # بهینه سازی هایپراپارامترها
                self.chatbot.knn = KNeighborsClassifier(
                    n_neighbors={n_neighbors}, 
                    weights='{weights}', 
                    metric='{metric}'
                )
                self.chatbot.mlp.hidden_layer_sizes = {hidden_layers}
                self.chatbot.mlp.activation = '{activation}'
                self.chatbot.train(self.chatbot.memory)
            """,
            'architecture': """
                # تغییر معماری مدل
                class ImprovedModel:
                    def __init__(self):
                        {new_model_code}
                self.chatbot.mlp = ImprovedModel()
                self.chatbot.train(self.chatbot.memory)
            """,
            'data_augmentation': """
                # افزودن داده‌های مصنوعی
                synthetic_data = {synthetic_data}
                self.chatbot.memory.extend(synthetic_data)
                self.chatbot.train(self.chatbot.memory)
            """
        }
        # انتخاب استراتژی بر اساس نوع مشکل
        return dedent(code_templates[problem_type])
    
    def _dynamic_execute(self, code):
        """اجرای امن کد تولید شده"""
        allowed_imports = {
            'sklearn.neighbors': ['KNeighborsClassifier'],
            'sklearn.neural_network': ['MLPClassifier'],
            'numpy': ['array', 'random']
        }
        
        # تحلیل امنیتی کد
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed_imports:
                        raise ImportError(f"Import {alias.name} not allowed!")
            elif isinstance(node, ast.ImportFrom):
                if node.module not in allowed_imports:
                    raise ImportError(f"Import from {node.module} not allowed!")
                for alias in node.names:
                    if alias.name not in allowed_imports[node.module]:
                        raise ImportError(f"Cannot import {alias.name} from {node.module}")

        # اجرای کد در محیط امن
        env = {
            'self': self.chatbot,
            '__builtins__': {**__builtins__, 'exec': None}
        }
        exec(code, env)
    
    def _optimize_hyperparameters(self):
        """بهینه سازی خودکار هایپراپارامترها"""
        code = self._generate_code('hyperparameters').format(
            n_neighbors=random.choice([3,5,7]),
            weights=random.choice(['uniform', 'distance']),
            metric=random.choice(['cosine', 'euclidean']),
            hidden_layers=random.choice([(256,128), (512,), (128,64)]),
            activation=random.choice(['relu', 'tanh'])
        )
        try:
            self._dynamic_execute(code)
            return True
        except Exception as e:
            self.error_log.append(f"Hyperparameter Error: {str(e)}")
            return False
    
    def _architectural_improvement(self):
        """بهبود معماری مدل با کد تولید شده"""
        new_model_code = """
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=(512, 256),
                activation='relu',
                learning_rate='adaptive',
                early_stopping=True
            )
        """
        code = self._generate_code('architecture').format(
            new_model_code=dedent(new_model_code)
        )
        try:
            self._dynamic_execute(code)
            return True
        except Exception as e:
            self.error_log.append(f"Architecture Error: {str(e)}")
            return False
    
    def _feature_engineering(self):
        """بهبود سیستم پیش پردازش متن"""
        new_preprocessing_code = """
            def _preprocess_text(self, text):
                text = text.lower().strip()
                text = re.sub(r'[^a-z0-9\\s]', '', text)
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(w) for w in tokens 
                        if w not in self.stop_words and len(w) > 2]
                return ' '.join(tokens)
        """
        try:
            self.chatbot._preprocess_text = types.MethodType(
                eval(new_preprocessing_code),
                self.chatbot
            )
            return True
        except Exception as e:
            self.error_log.append(f"Feature Engineering Error: {str(e)}")
            return False
    
    def _evaluate_performance(self):
        """ارزیابی عملکرد با داده های تست"""
        X = [item['user_input'] for item in self.chatbot.memory]
        y_true = [item['bot_response'] for item in self.chatbot.memory]
        
        if len(set(y_true)) < 2:
            return 0.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.2, random_state=42
        )
        
        # آموزش موقت
        
        temp_chatbot.train([{'user_input': x, 'bot_response': y} 
                          for x, y in zip(X_train, y_train)])
        
        # پیش‌بینی
        y_pred = [temp_chatbot.get_response(x) for x in X_test]
        
        return accuracy_score(y_test, y_pred)
    
    def _analyze_errors(self):
        """تحلیل خطاها و شناسایی الگوها"""
        error_patterns = Counter()
        for error in self.error_log:
            error_type = error.split(':')[0]
            error_patterns[error_type] += 1
        return error_patterns.most_common(1)
    
    def self_improve(self):
        """فرآیند اصلی خود-بهبودی"""
        prev_version = self.current_version
        new_version = f"{float(self.current_version)+0.1:.1f}"
        
        # ذخیره نسخه فعلی
        self._save_snapshot(prev_version)
        
        # انتخاب استراتژی بهبود
        current_accuracy = self._evaluate_performance()
        strategies = random.sample(self.improvement_strategies, 2)
        
        # اجرای استراتژی‌ها
        improvement_success = False
        for strategy in strategies:
            if strategy():
                improvement_success = True
                break
        
        # بررسی نتایج
        new_accuracy = self._evaluate_performance()
        if new_accuracy > current_accuracy + 0.1:
            self.current_version = new_version
            print(f"Successfully upgraded to v{new_version}!")
            return True
        else:
            print("Improvement failed. Rolling back...")
            self._rollback(prev_version)
            return False
    
    def auto_repair(self):
        """تعمیر خودکار بر اساس خطاهای ثبت شده"""
        common_error = self._analyze_errors()
        repair_strategies = {
            'Hyperparameter': self._optimize_hyperparameters,
            'Architecture': self._architectural_improvement,
            'Feature Engineering': self._feature_engineering
        }
        
        if common_error:
            error_type = common_error[0][0]
            strategy = repair_strategies.get(error_type, None)
            if strategy:
                return strategy()
        return False

class AdvancedChatbot:
    def __init__(self):
        self.memory = []
        self.vectorizer = TfidfVectorizer()
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            
        )
        self.logistic = LogisticRegression(solver='liblinear', random_state=42)
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_matrix = None
        self.context_matrix = None
        self.embedding_dim = 128
        self.response_cache = OrderedDict()
        self.cache_size = 50
        self.learning_rate = 0.01
        self.epochs = 300
        self.context_window = 3
        self.self_improver = SelfImprovingAI(self)

    # پیشپردازش پیشرفته با تصحیح املایی
    def _preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^a-z\s]', '', text)
        text = self._correct_spelling(text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return ' '.join(tokens)
        

    # تصحیح املایی با الگوریتم Damerau-Levenshtein
    
    
    def _correct_spelling(self, text, max_distance=2):
        words = text.split()
        corrected = []
        for word in words:
            if word in self.word_to_index:
                corrected.append(word)
                continue
            candidates = [(self._levenshtein_distance(word, w), w) 
                         for w in self.word_to_index if len(w) > 2]
            if candidates:
                closest = min(candidates)[1]
                corrected.append(closest)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def get_response(self, query):
        # چک کردن آیا مدل آموزش دیده یا نه
        if not hasattr(self.vectorizer, "vocabulary_"):
            return "I need training first! Please teach me something."
        
        if query in self.response_cache:
            self.response_cache.move_to_end(query)
            return self.response_cache[query]
        
        try:
            preprocessed = self._preprocess_text(query)
            input_vec = self.vectorizer.transform([preprocessed])
            input_scaled = self.scaler.transform(input_vec.toarray())
            
            responses = []
            for method in [self._knn_response, self._mlp_response, 
                          self._logistic_response, self._semantic_response]:
                resp = method(input_scaled if method != self._semantic_response else query)
                if resp:
                    responses.append(resp)
            
            final_response = self._ensemble_vote(responses) if responses else None
            
            if not final_response:
                sentiment = self._sentiment_analysis(query)
                final_response = self._natural_response(sentiment)
            
            if len(self.response_cache) >= self.cache_size:
                self.response_cache.popitem(last=False)
            self.response_cache[query] = final_response
            
            return final_response
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            return "Oops! I need more training to answer that."

    def interactive_learning(self, user_input, correct_response):
        """یادگیری از تعامل کاربر"""
        self.memory.append({
            'user_input': user_input,
            'bot_response': correct_response
        })
        self.self_improver.self_improve()

    def _levenshtein_distance(self, s, t):
        if len(s) < len(t):
            return self._levenshtein_distance(t, s)
        if len(t) == 0:
            return len(s)
        previous_row = range(len(t) + 1)
        for i, s_char in enumerate(s):
            current_row = [i + 1]
            for j, t_char in enumerate(t):
                cost = 0 if s_char == t_char else 1
                current_row.append(min(
                    current_row[j] + 1,
                    previous_row[j + 1] + 1,
                    previous_row[j] + cost
                ))
            previous_row = current_row
        return previous_row[-1]
    # تولید امبدینگ با الگوریتم GloVe
    def _generate_glove_embeddings(self):
        vocab_size = len(self.word_to_index)
        self.embedding_matrix = np.random.normal(size=(vocab_size, self.embedding_dim))
        self.context_matrix = np.random.normal(size=(vocab_size, self.embedding_dim))
        co_occurrence = defaultdict(lambda: defaultdict(float))
        for item in self.memory:
            words = self._preprocess_text(item['user_input']).split()
            for i, word in enumerate(words):
                if word not in self.word_to_index:
                    continue
                window = words[max(0, i-3):min(len(words), i+4)]
                for j, context_word in enumerate(window):
                    if i != j and context_word in self.word_to_index:
                       distance = abs(i - j)
                       weight = 1.0 / distance
                       cooccurrence[word][context_word] += weight

        for epoch in range(self.epochs):
            total_loss = 0
            for word, contexts in co_occurrence.items():
                word_idx = self.word_to_index[word]
                for context, count in contexts.items():
                    context_idx = self.word_to_index[context]
                    dot_product = np.dot(self.embedding_matrix[word_idx], 
                                      self.context_matrix[context_idx])
                    loss = dot_product - math.log(count + 1)
                    grad_word = loss * self.context_matrix[context_idx]
                    grad_context = loss * self.embedding_matrix[word_idx]
                    self.embedding_matrix[word_idx] -= self.learning_rate * grad_word
                    self.context_matrix[context_idx] -= self.learning_rate * grad_context
                    total_loss += 0.5 * loss**2
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
                

    # تشخیص احساسات کاربر
    def _sentiment_analysis(self, text):
        positive = {'happy', 'good', 'great', 'love', 'excellent', 'joy'}
        negative = {'sad', 'bad', 'hate', 'angry', 'terrible', 'worst'}
        tokens = self._preprocess_text(text).split()
        pos = sum(1 for w in tokens if w in positive)
        neg = sum(1 for w in tokens if w in negative)
        if pos > neg:
            return 'positive'
        elif neg > pos:
            return 'negative'
        else:
            return 'neutral'
     
    def _semantic_similarity(self, text1, text2):
        tokens1 = self._preprocess_text(text1).split()
        tokens2 = self._preprocess_text(text2).split()
        vec1 = self._get_embedding_vector(tokens1)
        vec2 = self._get_embedding_vector(tokens2)
        if vec1 is not None and vec2 is not None:
            return cosine_similarity([vec1], [vec2])[0][0]
        return 0.0
    def _get_embedding_vector(self, tokens):
        vectors = []
        for word in tokens:
            if word in self.word_to_index:
                vectors.append(self.embedding_matrix[self.word_to_index[word]])
        return np.mean(vectors, axis=0) if vectors else None
    def _natural_response(self, response_type):
        variants = {
            'greeting': ["Hi there!", "Hello!", "Hey! How can I help?", "Greetings!"],
            'farewell': ["Goodbye!", "See you later!", "Take care!", "Bye!"],
            'positive': ["That's great!", "Awesome!", "Glad to hear that!", "Wonderful!"],
            'negative': ["I'm sorry :(", "That must be hard", "Let's talk about it", "How can I help?"],
            'unknown': ["Hmm, can you explain more?", "Interesting, tell me more!", 
                       "I'm still learning, could you rephrase that?"]
        }
        return random.choice(variants[response_type])

    # سیستم رأیگیری بین مدل‌ها
    def _ensemble_vote(self, responses):
        votes = Counter(responses)
        top = votes.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1]:
            return top[0][0]
        return random.choice([top[0][0], top[1][0]])

    # تولید پاسخ از مدل‌های مختلف
    def _knn_response(self, input_vec):
        distances, indices = self.knn.kneighbors(input_vec)
        if distances[0][0] < 0.5:
            return self.memory[indices[0][0]]['bot_response']
        return None

    def _mlp_response(self, input_vec):
        if self.mlp.classes_.size == 0:
            return None
        proba = self.mlp.predict_proba(input_vec)[0]
        if np.max(proba) > 0.6:
            return self.label_encoder.inverse_transform([np.argmax(proba)])[0]
        return None

    def _logistic_response(self, input_vec):
        return self.logistic.predict(input_vec)[0]

    def _semantic_response(self, query):
        best_score, best_resp = 0, None
        for item in self.memory:
            score = self._semantic_similarity(query, item['user_input'])
            if score > best_score:
                best_score, best_resp = score, item['bot_response']
        return best_resp if best_score > 0.4 else None

    # پاسخ‌دهی اصلی با ترکیب همه روش‌ها
    def get_response(self, query):
        if query in self.response_cache:
            self.response_cache.move_to_end(query)
            return self.response_cache[query]
        
        preprocessed = self._preprocess_text(query)
        input_vec = self.vectorizer.transform([preprocessed])
        input_scaled = self.scaler.transform(input_vec.toarray())
        
        responses = []
        for method in [self._knn_response, self._mlp_response, 
                      self._logistic_response, self._semantic_response]:
            resp = method(input_scaled if method != self._semantic_response else query)
            if resp:
                responses.append(resp)
        
        final_response = self._ensemble_vote(responses) if responses else None
        
        if not final_response:
            sentiment = self._sentiment_analysis(query)
            final_response = self._natural_response(sentiment)
        
        # کش پاسخ
        if len(self.response_cache) >= self.cache_size:
            self.response_cache.popitem(last=False)
        self.response_cache[query] = final_response
        
        return final_response

    # آموزش مدل‌ها
    def train(self, data):
        self.memory.extend(data)
        # ساخت واژگان
        all_words = []
        
        if not data:
           raise ValueError("Training data cannot be empty!")
    
    # بررسی ساختار داده‌ها
        for item in data:
              if 'user_input' not in item or 'bot_response' not in item:
                  raise ValueError("Invalid training item structure!")
        
        for item in self.memory:
            words = self._preprocess_text(item['user_input']).split()
            all_words.extend(words)
        vocab = [word for word, count in Counter(all_words).items() if count > 1]
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        
        # تولید امبدینگ
        self.embedding_matrix = np.random.randn(len(vocab), self.embedding_dim) * 0.01
        self.context_matrix = np.random.randn(len(vocab), self.embedding_dim) * 0.01
        self._generate_glove_embeddings()
        
        # آموزش مدل‌های کلاسیک
        texts = [self._preprocess_text(item['user_input']) for item in self.memory]
        X = self.vectorizer.fit_transform(texts)
        X_scaled = self.scaler.fit_transform(X.toarray())
        y = self.label_encoder.fit_transform([item['bot_response'] for item in self.memory])
        
        self.knn.fit(X_scaled, y)
        self.mlp.fit(X_scaled, y)
        self.logistic.fit(X, y)
        
    
    def learn_from_user(self, user_input, correct_response):
        self.memory.append({'user_input': user_input, 'bot_response': correct_response})
        self.train(self.memory)  # بازآ
    # بررسی ساختار داده‌ها
    


    if __name__ == "__main__":
        bot = AdvancedChatbot()
    
        training_data = [
            {"user_input": "hello", "bot_response": "greeting"},
            {"user_input": "hi", "bot_response": "greeting"},
            {"user_input": "good morning", "bot_response": "greeting"},
            {"user_input": "goodbye", "bot_response": "farewell"},
            {"user_input": "see you", "bot_response": "farewell"},
            {"user_input": "bye", "bot_response": "farewell"}
    ]
    
        try:
            print("در حال آموزش مدل...")
            bot.train(training_data)
            print("آموزش موفقیت‌آمیز بود!")
        
            print("\nتست پاسخ‌ها:")
            print("کاربر: hello => ربات:", bot.get_response("hello"))
            print("کاربر: see you later => ربات:", bot.get_response("see you later"))
        
        except Exception as e:
            print(f"خطا در آموزش: {str(e)}")   