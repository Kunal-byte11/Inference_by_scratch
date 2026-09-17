import numpy as np
from layers import layer_norm
from transformers import GPT2Tokenizer


# ==========================================
# MODEL CONFIG
# ==========================================

VOCAB_SIZE = 50257
HIDDEN_SIZE = 768
MAX_SEQUENCE_LENGTH = 1024

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64


# ==========================================
# LOAD MODEL WEIGHTS
# ==========================================

WTE = np.loadtxt(
    "../weights/transformer.wte.weight.txt"
).reshape(VOCAB_SIZE, HIDDEN_SIZE)

WPE = np.loadtxt(
    "../weights/transformer.wpe.weight.txt"
).reshape(MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)


# ==========================================
# TOKENIZER
# ==========================================

tokenizer = GPT2Tokenizer.from_pretrained(
    "../weights/tokenizer"
)


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

def transformer_block(X, layer):

    # ======================================
    # 1. First LayerNorm
    # ======================================

    gamma = np.loadtxt(
        f"../weights/transformer.h.{layer}.ln_1.weight.txt"
    )

    beta = np.loadtxt(
        f"../weights/transformer.h.{layer}.ln_1.bias.txt"
    )

    X_norm = layer_norm(
        X,
        gamma,
        beta
    )


    # ======================================
    # 2. QKV Projection
    # ======================================

    att_weight = np.loadtxt(
        f"../weights/transformer.h.{layer}.attn.c_attn.weight.txt"
    ).reshape(
        HIDDEN_SIZE,
        3 * HIDDEN_SIZE
    )

    att_bias = np.loadtxt(
        f"../weights/transformer.h.{layer}.attn.c_attn.bias.txt"
    )

    QKV = X_norm @ att_weight + att_bias

    Q, K, V = np.split(
        QKV,
        3,
        axis=-1
    )


    # ======================================
    # 3. Split into Attention Heads
    # ======================================

    sequence_length = len(X)

    Q = Q.reshape(
        sequence_length,
        NUM_HEADS,
        HEAD_DIM
    )

    K = K.reshape(
        sequence_length,
        NUM_HEADS,
        HEAD_DIM
    )

    V = V.reshape(
        sequence_length,
        NUM_HEADS,
        HEAD_DIM
    )

    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)


    # ======================================
    # 4. Attention Scores
    # ======================================

    scores = Q @ K.transpose(0, 2, 1)

    scores = scores / np.sqrt(HEAD_DIM)


    # ======================================
    # 5. Causal Mask
    # ======================================

    mask = np.triu(
        np.ones(
            (sequence_length, sequence_length)
        ),
        k=1
    )

    scores = np.where(
        mask == 1,
        -np.inf,
        scores
    )


    # ======================================
    # 6. Softmax
    # ======================================

    scores = scores - np.max(
        scores,
        axis=-1,
        keepdims=True
    )

    exp_scores = np.exp(scores)

    attention_weights = (
        exp_scores
        /
        np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )
    )


    # ======================================
    # 7. Attention × V
    # ======================================

    attention_output = (
        attention_weights @ V
    )


    # ======================================
    # 8. Combine Heads
    # ======================================

    attention_output = attention_output.transpose(
        1,
        0,
        2
    )

    attention_output = attention_output.reshape(
        sequence_length,
        HIDDEN_SIZE
    )


    # ======================================
    # 9. Attention Output Projection
    # ======================================

    proj_weight = np.loadtxt(
        f"../weights/transformer.h.{layer}.attn.c_proj.weight.txt"
    ).reshape(
        HIDDEN_SIZE,
        HIDDEN_SIZE
    )

    proj_bias = np.loadtxt(
        f"../weights/transformer.h.{layer}.attn.c_proj.bias.txt"
    )

    attention_projected = (
        attention_output @ proj_weight
        + proj_bias
    )


    # ======================================
    # 10. Attention Residual
    # ======================================

    X = X + attention_projected


    # ======================================
    # 11. Second LayerNorm
    # ======================================

    gamma_2 = np.loadtxt(
        f"../weights/transformer.h.{layer}.ln_2.weight.txt"
    )

    beta_2 = np.loadtxt(
        f"../weights/transformer.h.{layer}.ln_2.bias.txt"
    )

    X_norm_2 = layer_norm(
        X,
        gamma_2,
        beta_2
    )


    # ======================================
    # 12. MLP First Projection
    # ======================================

    fc_weight = np.loadtxt(
        f"../weights/transformer.h.{layer}.mlp.c_fc.weight.txt"
    ).reshape(
        HIDDEN_SIZE,
        3072
    )

    fc_bias = np.loadtxt(
        f"../weights/transformer.h.{layer}.mlp.c_fc.bias.txt"
    )

    X_mlp = (
        X_norm_2 @ fc_weight
        + fc_bias
    )


    # ======================================
    # 13. GELU
    # ======================================

    X_mlp = 0.5 * X_mlp * (
        1
        +
        np.tanh(
            np.sqrt(2 / np.pi)
            *
            (
                X_mlp
                +
                0.044715 * X_mlp ** 3
            )
        )
    )


    # ======================================
    # 14. MLP Output Projection
    # ======================================

    mlp_proj_weight = np.loadtxt(
        f"../weights/transformer.h.{layer}.mlp.c_proj.weight.txt"
    ).reshape(
        3072,
        HIDDEN_SIZE
    )

    mlp_proj_bias = np.loadtxt(
        f"../weights/transformer.h.{layer}.mlp.c_proj.bias.txt"
    )

    mlp_output = (
        X_mlp @ mlp_proj_weight
        + mlp_proj_bias
    )


    # ======================================
    # 15. MLP Residual
    # ======================================

    X = X + mlp_output

    return X


# ==========================================
# FORWARD PASS
# ==========================================

def forward(tokens):

    # ======================================
    # 1. Token Embeddings
    # ======================================

    token_embeddings = WTE[tokens]


    # ======================================
    # 2. Position Embeddings
    # ======================================

    positions = np.arange(
        len(tokens)
    )

    position_embeddings = WPE[positions]


    # ======================================
    # 3. Combine Embeddings
    # ======================================

    X = (
        token_embeddings
        +
        position_embeddings
    )


    # ======================================
    # 4. Transformer Blocks
    # ======================================

    for layer in range(NUM_LAYERS):

        X = transformer_block(
            X,
            layer
        )


    # ======================================
    # 5. Final LayerNorm
    # ======================================

    final_gamma = np.loadtxt(
        "../weights/transformer.ln_f.weight.txt"
    )

    final_beta = np.loadtxt(
        "../weights/transformer.ln_f.bias.txt"
    )

    X = layer_norm(
        X,
        final_gamma,
        final_beta
    )


    # ======================================
    # 6. LM Head
    # ======================================

    logits = X @ WTE.T

    return logits


# ==========================================
# GENERATE
# ==========================================

def generate(
    tokens,
    max_new_tokens=50
):

    tokens = tokens.copy()

    for _ in range(max_new_tokens):

        # Keep only the latest 1024 tokens
        model_tokens = tokens[
            -MAX_SEQUENCE_LENGTH:
        ]

        # Forward pass
        logits = forward(
            model_tokens
        )

        # Last token's logits
        next_token_logits = logits[-1]

        # Greedy decoding
        next_token_id = int(
            np.argmax(next_token_logits)
        )

        # Add new token
        tokens.append(
            next_token_id
        )

    return tokens


# ==========================================
# CHAT
# ==========================================

def chat():

    conversation = ""

    print("\n================================")
    print("       NumPy GPT-2 Chat")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("Goodbye!")
            break


        # Add user message
        conversation += (
            f"User: {user_input}\n"
            f"Assistant:"
        )


        # Convert conversation to token IDs
        tokens = tokenizer.encode(
            conversation
        )


        # Generate response
        generated_tokens = generate(
            tokens,
            max_new_tokens=50
        )


        # Extract only newly generated tokens
        response_tokens = generated_tokens[
            len(tokens):
        ]


        # Decode response
        response = tokenizer.decode(
            response_tokens
        )


        # Print response
        print(
            f"GPT-2: {response}"
        )


        # Add response to conversation
        conversation += (
            f" {response}\n"
        )


# ==========================================
# START CHAT
# ==========================================

if __name__ == "__main__":
    chat()