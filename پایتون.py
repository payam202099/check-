import json
import requests
import os
import re
import hashlib
import numpy as np
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import json
import requests
import os
import re
import hashlib
import numpy as np
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
import time
import random
import string
from tqdm import tqdm
import logging
import base64
from collections import defaultdict, Counter
from functools import lru_cache
from itertools import chain
from sklearn.neighbors import NearestNeighbors  # Add this line
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
import time
import random
import string
from tqdm import tqdm
import logging
import base64
from collections import defaultdict, Counter
from functools import lru_cache
from itertools import chain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedSecurity:
    def __init__(self):
        self.cipher = Fernet.generate_key()
        self.hashing_iterations = 1000000  # Increased iterations for stronger hashing
        self.pepper = os.urandom(128)  # Significantly larger pepper
        self.salt = os.urandom(64)  # Added salt for password hashing
        self.hmac_key = os.urandom(32)  # Key for HMAC

    @staticmethod
    def generate_secure_key():
        return Fernet.generate_key()

    def secure_hash(self, data: str) -> str:
        """تابع هش مقاوم در برابر حملات با استفاده از SHA-512 و PBKDF2"""
        salted_data = data.encode() + self.salt + self.pepper
        key = hashlib.pbkdf2_hmac('sha512', salted_data, self.salt, self.hashing_iterations, dklen=64)
        return base64.urlsafe_b64encode(key).decode()

    def encrypt_data(self, data: str) -> bytes:
        """رمزنگاری پیشرفته با الگوریتم Fernet و اضافه کردن HMAC"""
        encrypted = Fernet(self.cipher).encrypt(data.encode())
        hmac_tag = self._compute_hmac(encrypted)
        return encrypted + hmac_tag

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """رمزگشایی ایمن با بررسی HMAC"""
        data_length = len(encrypted_data) - 32  # Assuming HMAC is 32 bytes (SHA-256)
        encrypted, hmac_tag = encrypted_data[:data_length], encrypted_data[data_length:]
        if not self._verify_hmac(encrypted, hmac_tag):
            raise ValueError("HMAC verification failed!")
        return Fernet(self.cipher).decrypt(encrypted).decode()

    def _compute_hmac(self, message: bytes) -> bytes:
        """Compute HMAC for integrity check"""
        return hashlib.hmac.new(self.hmac_key, message, hashlib.sha256).digest()

    def _verify_hmac(self, message: bytes, hmac_tag: bytes) -> bool:
        """Verify the HMAC tag"""
        computed_hmac = self._compute_hmac(message)
        return hmac.compare_digest(computed_hmac, hmac_tag)

    def secure_random(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return os.urandom(length)

class PersianNLPEngine:
    def __init__(self):
        self.stop_words = self._load_stopwords()
        self.vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2)
        self.svd = TruncatedSVD(n_components=1000)
        self.nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.stemmer = self._create_stemmer()
        self.lemmatizer = self._create_lemmatizer()

    def _load_stopwords(self) -> set:
        """بارگذاری لیست توقف‌واژه‌های فارسی با افزودن کلمات جدید"""
        return set(['و', 'در', 'به', 'از', 'را', 'که', 'با', 'این', 'هم', 'برای', 'بود', 'شد', 'است', 'زمانی', 'کرد'])

    def _create_stemmer(self):
        # Placeholder for stemmer creation; real implementation would use an actual Persian stemmer
        return lambda x: x

    def _create_lemmatizer(self):
        # Placeholder for lemmatizer; real implementation would use a Persian lemmatizer
        return lambda x: x

    @staticmethod
    def preprocess_text(text: str) -> str:
        """پیش‌پردازش متن فارسی با اعمال استمینگ و لماتیزیشن"""
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Persian characters
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.stemmer(word) for word in words]
        words = [self.lemmatizer(word) for word in words]
        return ' '.join(words)

    def build_semantic_index(self, documents: List[str]):
        """ساخت نمایه معنایی با استفاده از SVD و Nearest Neighbors"""
        cleaned = [self.preprocess_text(doc) for doc in documents]
        tfidf_matrix = self.vectorizer.fit_transform(cleaned)
        reduced = self.svd.fit_transform(tfidf_matrix)
        self.nn.fit(reduced)

    def semantic_search(self, query: str, k: int = 5) -> List[int]:
        """جستجوی معنایی با استفاده از Nearest Neighbors"""
        cleaned = self.preprocess_text(query)
        vec = self.vectorizer.transform([cleaned])
        reduced = self.svd.transform(vec)
        distances, indices = self.nn.kneighbors(reduced)
        return indices[0].tolist()

    def extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """استخراج کلمات کلیدی با استفاده از TF-IDF"""
        vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vec.toarray()[0]
        return [feature_names[i] for i in np.argsort(scores)[-n:] if scores[i] > 0]

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """تولید خلاصه متن با استفاده از روش TF-IDF و جملات با امتیاز بالا"""
        sentences = re.split(r'[.؟!]', text)
        sentence_vectors = self.vectorizer.transform(sentences)
        sentence_scores = np.sum(sentence_vectors.toarray(), axis=1)
        top_sentences = np.argsort(sentence_scores)[-num_sentences:][::-1]
        return ' '.join([sentences[i] for i in top_sentences])

class AutonomousLearner:
    def __init__(self, security: AdvancedSecurity):
        self.security = security
        self.nlp = PersianNLPEngine()
        self.knowledge_graph = {}
        self.executor = ThreadPoolExecutor(max_workers=16)  # Increased for higher concurrency
        self.cache = {}
        self.error_log = []

    def learn_from_web(self, base_url: str, depth: int = 4):
        """یادگیری خودکار از منابع وب با عمق بیشتر و کش"""
        visited = set()
        queue = [(base_url, 0)]
        
        with tqdm(total=100, desc="یادگیری از وب") as pbar:
            while queue:
                url, current_depth = queue.pop(0)
                if url in visited or current_depth > depth:
                    pbar.update(1)
                    continue
                
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        self.cache[url] = soup
                        paragraphs = [p.get_text() for p in soup.find_all('p')]
                        self._process_content(paragraphs)
                        
                        for link in soup.find_all('a', href=True):
                            absolute_url = urljoin(url, link['href'])
                            if self._is_valid_url(absolute_url) and absolute_url not in visited:
                                queue.append((absolute_url, current_depth + 1))
                        
                        visited.add(url)
                    else:
                        self.error_log.append(f"Failed to fetch {url}: {response.status_code}")
                except Exception as e:
                    self.error_log.append(f"Error processing {url}: {str(e)}")
                pbar.update(1)
        
        logging.info("Web learning completed with errors:")
        for error in self.error_log:
            logging.error(error)

    def _is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid for crawling"""
        parsed = urlparse(url)
        return bool(parsed.netloc) and not url.startswith('mailto:')

    def _process_content(self, paragraphs: List[str]):
        """پردازش محتوای استخراج شده با توجه به کیفیت و محتوا"""
        futures = []
        for para in paragraphs:
            if len(para.split()) > 20:  # Filter out very short paragraphs
                futures.append(self.executor.submit(
                    self._analyze_paragraph,
                    para
                ))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                self._update_knowledge_graph(*result)
            except Exception as e:
                logging.error(f"Error in processing paragraph: {str(e)}")

    def _analyze_paragraph(self, text: str) -> Tuple[str, Dict]:
        """تحلیل عمیق پاراگراف با رویکردهای متنوع"""
        cleaned = self.nlp.preprocess_text(text)
        keywords = self.nlp.extract_keywords(cleaned)
        entities = self._recognize_entities(cleaned)
        sentiment = self._analyze_sentiment(cleaned)
        return (cleaned, {'keywords': keywords, 'entities': entities, 'sentiment': sentiment})

    def _recognize_entities(self, text: str) -> Dict[str, List[str]]:
        """تشخیص موجودیت‌های نامدار با استفاده از روش‌های ساده‌تر"""
        entities = {
            'persons': re.findall(r'(?<=\b)[آ-ی]{3,}(?=\s)', text),
            'locations': re.findall(r'(?<=\b)\b[A-Z][a-z]+(?=\b)', text),
            'organizations': re.findall(r'(?<=\b)[آ-ی]{3,} (?=\b)', text),
            'dates': re.findall(r'\d{4}(?:[/]\d{2}){2}', text)  # Very basic date recognition
        }
        return {k: list(set(v)) for k, v in entities.items() if v}

    def _analyze_sentiment(self, text: str) -> str:
        """تحلیل احساسات به صورت ساده بر اساس کلمات - this is a very basic approach"""
        positive_words = ['خوب', 'عالی', 'موفق', 'بهترین', 'شادی']
        negative_words = ['بد', 'ناامید', 'شکست', 'غم', 'بدترین']
        sentiment_score = sum(1 for word in text.split() if word in positive_words) - \
                          sum(1 for word in text.split() if word in negative_words)
        return 'مثبت' if sentiment_score > 0 else 'منفی' if sentiment_score < 0 else 'خنثی'

    def _update_knowledge_graph(self, text: str, metadata: Dict):
        """به روزرسانی گراف دانش با اطلاعات مفصل‌تر"""
        doc_hash = self.security.secure_hash(text)
        self.knowledge_graph[doc_hash] = {
            'content': self.security.encrypt_data(text),
            'metadata': metadata,
            'timestamp': time.time()
        }

class AutoLearnerBot:
    def __init__(self):
        self.security = AdvancedSecurity()
        self.learner = AutonomousLearner(self.security)
        self.knowledge_base = {}
        self.learned_sources = set()
        self.file_path = "auto_knowledge.json"
        self.load_knowledge()
        self.search_engines = {
            'google': {'url': 'https://www.googleapis.com/customsearch/v1',
                       'params': {'key': "AIzaSyC2SXs0CvldTryUIfFpTtEXEu4VZliCfSk", 'cx': "4296cffda01e842f1", 'num': 10, 'lr': 'lang_fa', 'q': ''}},
            'wikipedia': {'url': 'https://fa.wikipedia.org/w/api.php',
                          'params': {'action': 'query', 'list': 'search', 'format': 'json', 'srsearch': ''}},
            'arxiv': {'url': 'http://export.arxiv.org/api/query',
                      'params': {'search_query': '', 'start': 0, 'max_results': 10}},
            'github': {'url': 'https://api.github.com/search/repositories',
                       'params': {'q': '', 'sort': 'updated'}},
            'stackoverflow': 'https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={}&site=stackoverflow',
            'technical_blogs': 'https://api.github.com/search/repositories?q={}+in:readme&sort=updated'
        }

    def load_knowledge(self):
        """Load existing knowledge base from JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            logging.warning("Knowledge base file not found. Starting with an empty base.")
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON in knowledge base. Starting with an empty base.")

    def save_knowledge(self):
        """Save the knowledge base to JSON file"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    def online_learn(self, query: str):
        """یادگیری خودکار از منابع آنلاین"""
        if query in self.knowledge_base:
            return
        
        print(f"🔎 در حال یادگیری '{query}' از اینترنت...")
        learned_data = []
        for source, config in self.search_engines.items():
            if isinstance(config, dict):  # For structured sources like Google, Wikipedia, etc.
                try:
                    config['params']['q' if 'q' in config['params'] else 'srsearch'] = query
                    response = requests.get(config['url'], params=config['params'])
                    response.raise_for_status()
                    data = response.json()
                    learned_data.extend(self._process_source(source, data))
                except requests.RequestException as e:
                    logging.error(f"Error fetching from {source}: {e}")
            else:  # For simple URL format like StackOverflow
                try:
                    url = config.format(quote_plus(query))
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    learned_data.extend(self._process_source(source, data))
                except requests.RequestException as e:
                    logging.error(f"Error fetching from {source}: {e}")

        if learned_data:
            self._extract_key_info(query, learned_data)
            self.save_knowledge()

    def _process_source(self, source: str, data: Dict[str, Any]) -> List[str]:
        """پردازش داده‌های دریافتی از منابع"""
        processed = []
        
        if source == 'google':
            for item in data.get('items', []):
                processed.append(item.get('snippet', ''))
        elif source == 'wikipedia':
            for item in data.get('query', {}).get('search', [])[:3]:
                page_url = f"https://fa.wikipedia.org/?curid={item['pageid']}"
                content = self._scrape_page(page_url)
                processed.append(content)
        elif source == 'arxiv':
            for entry in data.get('feed', {}).get('entry', []):
                processed.append(entry.get('summary', ''))
        elif source == 'github':
            for repo in data.get('items', []):
                readme_url = f"https://raw.githubusercontent.com/{repo['full_name']}/master/README.md"
                content = self._scrape_page(readme_url, is_markdown=True)
                processed.append(content)
        elif source == 'stackoverflow':
            for item in data.get('items', []):
                processed.append(f"{item['title']}\n{self._clean_html(item['body'])}")
        elif source == 'technical_blogs':
            for repo in data.get('items', []):
                content = self._scrape_page(f"https://raw.githubusercontent.com/{repo['full_name']}/master/README.md", is_markdown=True)
                processed.append(content)

        return processed

    def _scrape_page(self, url: str, is_markdown: bool = False) -> str:
        """استخراج محتوای اصلی از صفحات وب"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if is_markdown:
                return self._clean_markdown(response.text)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in ['script', 'style', 'nav', 'footer']:
                for element in soup.select(tag):
                    element.decompose()
            return ' '.join([p.get_text().strip() for p in soup.select('p, h1, h2, h3') if p.get_text().strip()])
        except requests.RequestException as e:
            logging.error(f"Error scraping page {url}: {e}")
            return ""

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

    def _clean_markdown(self, markdown_content: str) -> str:
        """Clean markdown content to plain text"""
        # This is a very basic approach, real markdown parsing would need a dedicated parser
        return re.sub(r'#+|`|\*|_|~|-|\[|\]|>|!|\(|\)', '', markdown_content)

    def _extract_key_info(self, query: str, texts: List[str]):
        """استخراج اطلاعات کلیدی با الگوریتم ساده"""
        all_words = chain.from_iterable(re.findall(r'\w+', text.lower()) for text in texts)
        word_counts = Counter(all_words)
        top_keywords = [word for word, _ in word_counts.most_common(10) if word not in self.learner.nlp.stop_words]

        summary = self.learner.nlp.summarize(' '.join(texts), num_sentences=3)
        
        self.knowledge_base[query] = {
            'keywords': top_keywords,
            'summary': summary,
            'sources': texts[:3]
        }

    def answer_question(self, question: str) -> str:
        """پاسخ به سوالات با استفاده از دانش آموخته شده"""
        if question in self.knowledge_base:
            return self.knowledge_base[question]['summary']
        
        # If exact match not found, look through keywords
        question_keywords = set(re.findall(r'\w{3,}', question.lower()))
        for topic, data in self.knowledge_base.items():
            if len(question_keywords & set(data['keywords'])) >= 3:
                return data['summary']
        
        # If no match found, attempt to learn
        self.online_learn(question)
        return self.knowledge_base.get(question, {}).get('summary', "هنوز یاد نگرفتم، ولی بعدا میدونم!")

# Main loop to simulate bot interaction
bot = AutoLearnerBot()
while True:
    user_input = input("شما: ")
    if user_input.lower() == 'خروج':
        break
    response = bot.answer_question(user_input)
    print("چتبات:", response)