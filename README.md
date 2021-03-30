# RNA-DLM
A deep neural network to learn the RNA language model. It uses [SentencePiece](https://github.com/google/sentencepiece) to tokenize the RNA sequence and a Transformer Encoder network trained with masked language modeling. Hopefully, this network pre-trained in an unsupervised way could be used in many sequence classification problems in Bioinformatics.

Unfortunately, I do not have the computation power needed to train it. I have tried a 1080 ti, but after weeks the loss only decreases a little.
