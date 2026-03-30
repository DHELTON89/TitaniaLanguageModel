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

main()