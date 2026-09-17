import numpy as np

def softmax(x):
    x = x- np.max(x , axis = -1, keepdims= True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x , axis = -1 , keepdims = True)


def causal_attention(Q,K,V):
    # head dimension

    d_k = Q.shape[-1]

    # Calculate attention scores

    scores = Q @ K.T

    # Scale scores

    scores = scores / np.sqrt(d_k)

    # Create causal mask

    sequence_length = Q.shape[0]

    mask = np.triu(

        np.ones((sequence_length,sequence_length)),

        k = 1

    )

    scores = np.where(

        mask == 1 ,
        -np.inf,
        scores
    )


    # Convert scores to probabilities

    attention_weights = softmax(scores)

    # weighted sum of values

    output = attention_weights @ V

    return output