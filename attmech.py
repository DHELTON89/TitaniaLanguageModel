#This code runs an attention mechanism
#There are four main types of attention mechanism
#1. simplified self-attention: introduces the broader idea
#2. self-attention: uses trainable weights that are the primary
#idea behind LLMs.
#3. causal attention: model only consideers previous and current
#inputs in sequence
#4. multi-head attention: model retrieves information from different
#subspaces

import torch
#This converts specifics words into three-dimensional vectors
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your (x^1)
    [0.55, 0.87, 0.66], #journey (x^2)
    [0.57, 0.85, 0.64], #starts (x^3)
    [0.22, 0.58, 0.33], #with (x^4)
    [0.77, 0.25, 0.10], #one (x^5)
    [0.05, 0.80, 0.55]] #step (x^6)
)
#Notice how if each second decimal is ignored, then the 
#vectors for 'journey' and 'starts' are the exact same.

def main():
    #print("This code runs")
#from the line under import torch above, this code will
#calculate intermediate attention scores  between the 
#query token and each input token
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
#Now normalize each of the attention scores
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
#The attention scores printed out earlier should now sum to 1

#The softmax function can perform the same role, but is more
#efficient. 
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum(softmax):", attn_weights_2_naive.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    print("Context Vector Total")
#Now we can calculate the context vector
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i
    print(context_vec_2)
#This is the weighted sum of all context vectors
#It should appear as just one line below context vector total

#Now we compute all attention weights and not just the second one
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)
#This will print out ([0.9995, 0.9544, 0.9422,...,0.9450])

#After this is done, we add trainable weights, which will allow the language
#model to learn from data and improve performance on specific tasks.
#Three weight matrices must be initialized
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
#requires_grad is set to False to reduce noise from the outputs, but in actual
#model training it would be set to True so the matrices can update in real time.
#now compute the query, key and value vectors; two numbers should print out (0.4306
# and 1.4551)
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_query
    value_2 = x_2 @ W_value
    print("Now we are adding trainable weights")
    print(query_2)
#A two-dimensional vector will print out, since the number of columns of the 
#corresponding weight matrix is set by d_out = 2

#You can obtain all keys and values via matrix multiplication
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
#Now compute the attention score
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)

    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)
#Now we go from attention scores to attention weights. Compute the attention
#weights by scaling the attention scores and using the softmax function. 
#However, now we scale the attention scores by dividing them by the square
#root is mathematically the same as exponentiating by 0.5
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim =  -1)
    print(attn_weights_2)
#Now the final step is to compute the context vectors
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)
    
main()