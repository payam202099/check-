
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial
import optax


# 1. Data Generation and Preprocessing
def generate_data(num_samples=100000):
    expressions = []
    results = []
    for _ in range(num_samples):
        num1 = np.random.randint(1, 100)
        num2 = np.random.randint(1, 100)
        op = np.random.choice(['+', '-', '*', '/'])
        
        if op == '/':
             num1 = num1 * num2 # Ensure integer division
        
        expr = f"{num1}{op}{num2}"
        try:
            result = eval(expr)
            expressions.append(expr)
            results.append(result)
        except Exception:
            continue  # Skip invalid expressions
    return pd.DataFrame({'expression': expressions, 'result': results})

def build_vocabulary(df):
    tokens = set()
    for expr in df['expression']:
        tokens.update(list(expr))
    tokens = sorted(list(tokens))
    tokens = ['[PAD]', '[UNK]'] + tokens
    token_to_id = {token: i for i, token in enumerate(tokens)}
    return token_to_id

def numericalize_expression(expr, token_to_id):
    return [token_to_id.get(token, token_to_id['[UNK]']) for token in expr]


# 2. Transformer Model with Jax
class TransformerMathSolver:
    def __init__(self, vocab_size, max_len, d_model=256, n_head=8, num_layers=6, d_ff=1024, dropout_rate=0.1):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Initialize weights using Jax
        self.params = self._init_params(jax.random.PRNGKey(0))
        

    def _init_params(self, key):
        keys = jax.random.split(key, 8)
        params = {
            'embedding': jax.random.normal(keys[0], (self.vocab_size, self.d_model)) * 0.02,
            'pos_embedding': jax.random.normal(keys[1], (self.max_len, self.d_model)) * 0.02,
            'transformer_layers': [
                {
                    'attention': {
                        'W_q': jax.random.normal(keys[2], (self.d_model, self.d_model)) * 0.02,
                        'W_k': jax.random.normal(keys[3], (self.d_model, self.d_model)) * 0.02,
                        'W_v': jax.random.normal(keys[4], (self.d_model, self.d_model)) * 0.02,
                        'W_o': jax.random.normal(keys[5], (self.d_model, self.d_model)) * 0.02
                    },
                    'feed_forward': {
                        'W_1': jax.random.normal(keys[6], (self.d_model, self.d_ff)) * 0.02,
                        'W_2': jax.random.normal(keys[7], (self.d_ff, self.d_model)) * 0.02,
                        'b_1': jnp.zeros((self.d_ff,)),
                        'b_2': jnp.zeros((self.d_model,))
                    },
                    'layer_norm_1_gamma': jnp.ones(self.d_model),
                    'layer_norm_1_beta': jnp.zeros(self.d_model),
                     'layer_norm_2_gamma': jnp.ones(self.d_model),
                     'layer_norm_2_beta': jnp.zeros(self.d_model)
                }
                for _ in range(self.num_layers)
            ],
            'W_out': jax.random.normal(keys[0], (self.d_model, 1)) * 0.02,
            'b_out': jnp.zeros((1,))
            
        }
        return params
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(self.d_model)
        if mask is not None:
            attn_scores = jnp.where(mask, attn_scores, -1e9)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        return jnp.matmul(attn_weights, v)

    def multi_head_attention(self, x, params, mask=None):
        q = jnp.matmul(x, params['W_q'])
        k = jnp.matmul(x, params['W_k'])
        v = jnp.matmul(x, params['W_v'])

        batch_size = x.shape[0]
        q = jnp.reshape(q, (batch_size, -1, self.n_head, self.d_model // self.n_head))
        k = jnp.reshape(k, (batch_size, -1, self.n_head, self.d_model // self.n_head))
        v = jnp.reshape(v, (batch_size, -1, self.n_head, self.d_model // self.n_head))
        
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = jnp.swapaxes(attn_output, 1, 2)
        attn_output = jnp.reshape(attn_output, (batch_size, -1, self.d_model))

        return jnp.matmul(attn_output, params['W_o'])

    def feed_forward(self, x, params):
      x = jnp.matmul(x, params['W_1']) + params['b_1']
      x = jax.nn.relu(x)
      x = jnp.matmul(x, params['W_2']) + params['b_2']
      return x
    
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta

    def transformer_layer(self, x, params, mask=None):
        attn_output = self.multi_head_attention(x, params['attention'], mask)
        x = self.layer_norm(x + attn_output, params['layer_norm_1_gamma'], params['layer_norm_1_beta'])

        ff_output = self.feed_forward(x, params['feed_forward'])
        x = self.layer_norm(x+ff_output, params['layer_norm_2_gamma'], params['layer_norm_2_beta'])
        return x
    
    def compute_mask(self, x):
        mask = (x != 0).astype(jnp.float32)
        mask = mask[:, None, :]  # Add extra dimension
        return mask

    def forward(self, X, params, is_training=True):
        
        mask = self.compute_mask(X)

        x = params['embedding'][X]
        x += params['pos_embedding'][:X.shape[1]]

        for layer_params in params['transformer_layers']:
            x = self.transformer_layer(x, layer_params, mask=mask)
            
        x = jnp.mean(x, axis=1)

        return jnp.matmul(x, params['W_out']) + params['b_out']
    
    def compute_loss(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred)**2)
    
    
    def train_step(self, params, X, y, optimizer, opt_state):
        
      def loss_fn(params):
        y_pred = self.forward(X, params)
        loss = self.compute_loss(y, y_pred)
        return loss
        
      grad = jax.grad(loss_fn)(params)
      updates, opt_state = optimizer.update(grad, opt_state)
      new_params = optax.apply_updates(params, updates)
      loss = loss_fn(params)
      return new_params, opt_state, loss
    

# 3. Training Pipeline
def train_model():
    # Generate and prepare data
    df = generate_data(100000)
    token_to_id = build_vocabulary(df)
    vocab_size = len(token_to_id)
    
    # Convert expressions to numerical tokens
    X = np.array([numericalize_expression(expr, token_to_id) for expr in df['expression']])
    y = df['result'].values.reshape(-1, 1).astype(np.float32)
    
    # Pad sequences
    max_len = max(len(seq) for seq in X)
    X = np.array([seq + [token_to_id['[PAD]']]*(max_len-len(seq)) for seq in X])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = TransformerMathSolver(vocab_size=vocab_size, max_len=max_len)
    
    # Training parameters
    batch_size = 64
    epochs = 10000
    learning_rate = 0.1
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model.params)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = jnp.array(X_train[i:i+batch_size])
            y_batch = jnp.array(y_train[i:i+batch_size])
            
            model.params, opt_state, loss = model.train_step(model.params, X_batch, y_batch, optimizer, opt_state)

            epoch_loss += loss
        
        # Validation
        val_pred = model.forward(jnp.array(X_val), model.params, is_training=False)
        val_loss = model.compute_loss(jnp.array(y_val), val_pred)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss/len(X_train):.4f} | Val Loss: {val_loss:.4f}")
        
        # Learning rate decay
        if (epoch+1) % 20 == 0:
            learning_rate *= 0.5
            optimizer = optax.adam(learning_rate)
            print(f"Learning rate reduced to {learning_rate}")
    
    return model, token_to_id

# 4. Evaluation and Inference
def evaluate_model(model, token_to_id, num_tests=20):
    for _ in range(num_tests):
        num1 = np.random.randint(1, 100)
        num2 = np.random.randint(1, 100)
        op = np.random.choice(['+', '-', '*', '/'])
        
        if op == '/':
            num1 = num1 * num2  # Ensure integer division
            
        expr = f"{num1}{op}{num2}"
        tokens = numericalize_expression(expr, token_to_id)
        padded = tokens + [token_to_id['[PAD]']]*(model.max_len - len(tokens))
        
        prediction = model.forward(jnp.array([padded]), model.params, is_training=False)[0][0]
        actual = eval(expr)
        
        print(f"{expr} = {prediction:.2f} (True: {actual}) | Error: {abs(prediction-actual):.2f}")

# Run training and evaluation
if __name__ == "__main__":
    trained_model, token_to_id = train_model()
    evaluate_model(trained_model, token_to_id)
