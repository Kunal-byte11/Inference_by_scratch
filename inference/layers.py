import numpy as np

def layer_norm(x , gamma , beta , epsilon = 1e-5):
    mean = np.mean(x , axis = 1 , keepdims = True)

    variance = np.mean(

        (x - mean) ** 2,
        axis = -1,
        keepdims=True
    )

    normalized = (x - mean) / np.sqrt(variance + epsilon)

    output = gamma * normalized + beta

    return output


# # test 

# x = np.array([[10.0,20.0,30.0,40.0]])

# gamma = np.ones(4)

# beta = np.ones(4)

# output = layer_norm(x,gamma,beta)

# print(output)