#This imports a short story for training purposes
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
#This splits text
import re
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

def main():
    """
    print("Titania Initialization")
    print("Why isn't git working?")
if __name__ == "__main__":
    """
#An embedding is essentially a mapping of discrete objects to
#points in a continuous vector space, basically they convert 
#nonnumeric data into a format that neural networks can
#process. Words, sentences, and even paragraphs are tokenized and 
#given a value that corresponds to a specific number.

#Here we download text from the verdict

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

if __name__ == "__main__":
    main()
    