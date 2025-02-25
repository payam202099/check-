import jax
import jax.numpy as jnp
import numpy as np
import re
import optax
from flax import linen as nn
from flax.training import train_state
from typing import Any, Sequence
from sklearn.model_selection import train_test_split

# 1. Data Generation and Preprocessing
def generate_data(num_samples=100000):
    data = []
    operators = ['+', '-', '*', '/']
    for _ in range(num_samples):
        num1 = np.random.randint(1, 100)
        num2 = np.random.randint(1, 100)
        op = np.random.choice(operators)
        
        # Handle division cases
        if op == '/':
            num2 = max(1, num2)
            num1 = num1 * num2  # Ensure integer results
            
        expr = f"{num1}{op}{num2}"
        try:
            result = eval(expr)
            data.append([expr, float(result)])
        except:
            continue
    
    # Return as numpy array
    return np.array(data, dtype=object)

def tokenize_expression(expr):
    return re.findall(r'(\d+|[\+\-\*\/])', expr)

def build_vocab(expressions):
    all_tokens = set()
    for expr in expressions:
        tokens = tokenize_expression(expr)
        all_tokens.update(tokens)
    token_to_id = {t: i for i, t in enumerate(sorted(all_tokens))}
    token_to_id['[PAD]'] = len(token_to_id)
    return token_to_id

def process_data(data, token_to_id):
    expressions = data[:, 0]
    results = data[:, 1].astype(float)
    
    sequences = []
    max_len = 0
    for expr in expressions:
        tokens = tokenize_expression(expr)
        max_len = max(max_len, len(tokens))
        seq = [token_to_id[t] for t in tokens]
        sequences.append(seq)
    
    # Padding sequences
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = seq + [token_to_id['[PAD]']] * pad_len
        padded.append(padded_seq)
    
    return jnp.array(padded), jnp.array(results)

# 2. Model Definition with Flax
class MathTransformer(nn.Module):
    vocab_size: int
    max_len: int
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    
    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.d_model)
        self.pos_embed = nn.Embed(self.max_len, self.d_model)
        self.transformer = nn.Transformer(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            deterministic=True
        )
        self.output_proj = nn.Dense(1)
        
    def __call__(self, inputs):
        # Embeddings
        token_emb = self.embed(inputs)
        positions = jnp.arange(inputs.shape[-1])
        pos_emb = self.pos_embed(positions)
        x = token_emb + pos_emb
        
        # Transformer
        x = self.transformer(x)
        
        # Pooling and output
        x = jnp.mean(x, axis=1)
        return self.output_proj(x)

# 3. Training Utilities
def create_train_state(rng, vocab_size, max_len, learning_rate=0.001):
    model = MathTransformer(vocab_size=vocab_size, max_len=max_len)
    params = model.init(rng, jnp.ones((1, max_len), dtype=jnp.int32))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        inputs, targets = batch
        preds = state.apply_fn({'params': params}, inputs)
        loss = jnp.mean((preds.squeeze() - targets) ** 2)
        return loss, preds
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, batch):
    inputs, targets = batch
    preds = state.apply_fn({'params': state.params}, inputs)
    return jnp.mean((preds.squeeze() - targets) ** 2)

# 4. Main Training Loop
def main():
    # Generate data
    data = generate_data(10000)
    expressions = data[:, 0]
    
    # Build vocabulary
    token_to_id = build_vocab(expressions)
    vocab_size = len(token_to_id)
    
    # Process data
    X, y = process_data(data, token_to_id)
    max_len = X.shape[1]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create train state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, vocab_size, max_len)
    
    # Training parameters
    batch_size = 32
    epochs = 100
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        # Batch training
        epoch_loss = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            state, loss = train_step(state, (batch_X, batch_y))
            epoch_loss.append(loss)
        
        # Validation
        val_loss = eval_step(state, (X_val, y_val))
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {np.mean(epoch_loss):.4f} | Val Loss: {val_loss:.4f}")
    
    # Test some examples
    test_exprs = [
        "25*43", "100/4", "75-32", 
        "12+89", "64/8", "99*99"
    ]
    
    for expr in test_exprs:
        tokens = tokenize_expression(expr)
        seq = [token_to_id[t] for t in tokens]
        padded = seq + [token_to_id['[PAD]']] * (max_len - len(seq))
        inputs = jnp.array([padded])
        pred = state.apply_fn({'params': state.params}, inputs)
        actual = eval(expr)
        print(f"{expr} => Pred: {pred[0][0]:.2f}, Actual: {actual}")

if __name__ == "__main__":
    main()