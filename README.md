#Titania Language Model
# TitaniaLanguageModel
Simple Language Model

What is a language model?
A language model is a deep neural network that pushes out text or audio in response to an inquiry. It is a form of artificial intelligence.

This model is based on the book "Build a Large Language Model" by Sebastian Raschka.

ChatGPT was only used for fixing compiler errors and minor syntax issues.

This model is meant to perform simple tasks and engage in simple conversations.

This model consists of transformer architecture, notably using an encoder and a decoder module. The encoder processes input text and encodes it into numerical vectors. The decoder will work to generate (output) text based on those vectors. They are connected by something called a self-attention mechanism.

This mechanism weighs the importance of words (encodings) in relation to each other. The main.py section is essentially a tutorial on how language models work.

Use setup.py to understand how language models work. The code is explained in a way that is easy to understand and implement.

attmech.py (attention mechanism) further clarifies how the model should work