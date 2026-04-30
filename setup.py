#This imports a short story for training purposes
import urllib.request

#This splits text
import re
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
import torch
from torch.utils.data import Dataset, DataLoader

#This tokenizer handles word input and processing
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed
                        if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)
        return text

#This class handles unknown words not previously introduced
#from training sources
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',
                                 text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                    else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', 
                      r'\1', text)
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) 

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length + 1]
            target_chunk = token_ids[i + i: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
          
def main():
#An embedding is essentially a mapping of discrete objects to
#points in a continuous vector space, basically they convert 
#nonnumeric data into a format that neural networks can
#process. Words, sentences, and even paragraphs are tokenized and 
#given a value that corresponds to a specific number.

#Here we download text from the verdict
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

#This prints out the total number of characters and the first 99
# characters in the story. 

#The following snippet of code will split the text into whitespace 
#letters
    text = "Hello, world. This, is a test."
    result = re.split(r'(\s)', text)
    print(result)
#This next snippet of code will also include punctuation marks
    result = re.split(r'([,.]|\s)', text)
    print(result)
#Now we can remove whitespace characters here
    result = [item for item in result if item.strip()]
    print(result)
#Anything within the brackets will be separated and treated as it's
#own character ([.,!?_-])
    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)
#Every word and punctuation mark has been successfully tokenized

#Now start be tokenizing the verdict
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if 
                    item.strip()]
    print(len(preprocessed))
#Prints the number of tokens
    print(preprocessed[:30])
#prints the first 30 tokens

#if successful you should be able to convert a python string
#into an integer representation
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
#This should print out the number 1130
#Afterwords, print the first 51 entries
    vocab = {token:integer for integer, token in 
             enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
#If successful, then each token now has an integer representation
#9 = ;, 20 = Begin, 48 = He, 49 = Her
#Now we are ready to create tokenizer classes that will encode
#and decode text into integers and back into text
#Please refer to the SimpleTokenizerV1 class at the top of the
#code

#Apply a tokenizer and tokenize part of the verdict
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride"""
    ids = tokenizer.encode(text)
    print(ids)

#A list of numbers, [1, 56, 2, 850, 988...], should appear
#Then, print it back with the decode method
    print(tokenizer.decode(ids))

#If you want, uncomment and run this code. Since Hello is not used 
# in the verdict story, you should see a KeyError: 'Hello' appear, 
# since this word cannot be tokenized
    """
    text = "Hello, do you like tea?"
    print(tokenizer.encode(text))
    """
#Simple context tokens need to be implemented in order to 
#handle unknown words
#An <unk> token is not a part of the usual vocabulary
#An <endoftext> token separates unrelated text sources
    
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer, token in enumerate
             (all_tokens)}
    print(len(vocab.items()))
#It should say 1132, to show the new words introduced

#Use this text to test the SimpleTokenizerV2 class
    text1 = "Hello, do you like tea?"
    text2 = "in the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
#It should say "Hello, do  you like tea? <|endoftext|>
#in the sunlit terraces of the palace."

#Use this to encode the above text
    tokenizer  = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text))
#This will print out a new set of numbers

#This decodes it back
    print(tokenizer.decode(tokenizer.encode(text)))
    
#instatiate the BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    text = (
        "Hello, do you like tea?<|endoftext|>In the sunlit" \
        " terraces" " of someunknownplace.")
    integers = tokenizer.encode(text, allowed_special={
        "<|endoftext|>"})
    print (integers)
#This should print out a list of numbers[15496, 11, 466...]

#Now convert it back into a string
    strings = tokenizer.decode(integers)
    print(strings)
#You'll notice that someunknownplace is one word
#This is the result of Byte-pair encoding, which can handle
#unknown words

#implement a data loader that fetches input-target pairs
#basically the llm will predict the next word in a sequence
    with open("the-verdict.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

#Remove the first 50 tokens from the data set
    enc_sample = enc_text[50:]
#Context size determines how many tokens are included in the input
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size]
    print(f"x: {x}")
    print(f"y:     {y}")
#next word prediction:
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
#The code that prints out will look like this
# [input]---->predicted token 
#This will be in the form of integers

#Now look at the words, not the numbers
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode
            ([desired]))
        
#Create a dataset to load inputs into the DataLoader
    def create_dataloader_v1(txt, batch_size=4, max_length = 256,
                             stride = 128, shuffle = True, 
                             drop_last = True, num_workers = 0):
         tokenizer = tiktoken.get_encoding("gpt2")
         dataset = GPTDatasetV1(txt, tokenizer, max_length,
                                stride)
         dataloader = DataLoader(
             dataset,
             batch_size=batch_size,
             shuffle=shuffle,
             drop_last=drop_last,
             num_workers=num_workers)
         return dataloader

#Test the dataloader, you should see [tensor(some numbers)...]
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1,
        shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
#Second batch
    #second_batch = next(data_iter)
    #print(second_batch)

#If successful, try with a batch size greater than 1
    """
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4,
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    """
#This should print a much larger set of batches(tensor([[numbers]]))

#The final step is to convert token IDs into embedding vectors
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

#Apply a token ID to embedding vector
    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))

#Create a more realistic sized embedding
    vocab_size = 50527
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size,
                                               output_dim)
    
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

#Now apply the embedding layer to token IDs
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length,
                                             output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange
                                         (context_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

   

if __name__ == "__main__":
    main()
    