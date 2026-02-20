#Titania Language Model
# TitaniaLanguageModel
Simple Language Model

What is a language model?
A language model is is a deep neural network that pushes out text or audio in response to an inquiry. It is a form of artificial intelligence

This model is based on the book "Build a Large Language Model" by Sebastian Raschka

This model is meant to perform simple tasks and engage in simple conversations

This model consists of transformer architecture, notably using an encoder and a decoder module. The encoder processes input text and encodes it into numerical vectors. The decoder will work to generate (output) text based on those vectors. They are connected by something called a self-attention mechanism.

This mechanism weighs the importance of words (encodings) in relation to each other. This model is trained on 