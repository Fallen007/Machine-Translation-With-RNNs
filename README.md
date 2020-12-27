# Machine Translation with RNNs

## Goal

In this project, we build a deep neural network that functions as part of a machine translation pipeline. The pipeline accepts English text as input and returns the French translation. The goal is to achieve the highest translation accuracy possible.

##### &nbsp;

## Approach

To translate a corpus of English text to French, we need to build a recurrent neural network (RNN). Lets dive into the implementation.

##### &nbsp;

### Building the Pipeline

Below is a summary of the various preprocessing and modeling steps. The high-level steps include:

1. **Preprocessing**: load and examine data, cleaning, tokenization, padding
1. **Modeling**: build, train, and test the model
1. **Prediction**: generate specific translations of English to French, and compare the output translations to the ground truth translations
1. **Iteration**: iterate on the model, experimenting with different architectures

For a more detailed walkthrough including the source code, check out the Jupyter notebook in the main directory ([Machine_Translation_with RNNs.ipynb](Machine_Translation_with RNNs.ipynb)).

##### &nbsp;

### Toolset

We will be using Keras for this project.

##### &nbsp;

## Preprocessing

### Load & Examine Data

Here is a sample of the data. The inputs are sentences in English; the outputs are the corresponding translations in French.

> <img src="images/training-sample.png" width="100%" align="top-left" alt="" title="Data Sample" />

##### &nbsp;

When we run a word count, we can see that the vocabulary for the dataset is quite small. This allows us to train the models in a reasonable time.

> <img src="images/vocab.png" width="75%" align="top-left" alt="" title="Word count" />

### Cleaning

No additional cleaning needs to be done at this point. The data has already been converted to lowercase and split so that there are spaces between all words and punctuation.

_Note:_ For other NLP projects we may need to perform additional steps such as: remove HTML tags, remove stop words, remove punctuation or convert to tag representations, label the parts of speech, or perform entity extraction.

### Tokenization

Next we need to tokenize the data&mdash;i.e., convert the text to numerical values. This allows the neural network to perform operations on the input data. For this project, each word and punctuation mark will be given a unique ID. (For other NLP projects, it might make sense to assign each character a unique ID.)

When we run the tokenizer, it creates a word index, which is then used to convert each sentence to a vector.

> <img src="images/tokenizer.png" width="100%" align="top-left" alt="" title="Tokenizer output" />

### Padding

When we feed our sequences of word IDs into the model, each sequence needs to be the same length. To achieve this, padding is added to any sequence that is shorter than the max length (i.e. shorter than the longest sentence).

> <img src="images/padding.png" width="50%" align="top-left" alt="" title="Tokenizer output" />

## Modeling

First, let's breakdown the architecture of a RNN at a high level:

1. **Inputs** &mdash; Input sequences are fed into the model with one word for every time step. Each word is encoded as a unique integer or one-hot encoded vector that maps to the English dataset vocabulary.
1. **Embedding Layers** &mdash; Embeddings are used to convert each word to a vector. The size of the vector depends on the complexity of the vocabulary.
1. **Recurrent Layers (Encoder)** &mdash; This is where the context from word vectors in previous time steps is applied to the current word vector.
1. **Dense Layers (Decoder)** &mdash; These are typical fully connected layers used to decode the encoded input into the correct translation sequence.
1. **Outputs** &mdash; The outputs are returned as a sequence of integers or one-hot encoded vectors which can then be mapped to the French dataset vocabulary.

##### &nbsp;

### Embeddings

Embeddings allow us to capture more precise syntactic and semantic word relationships. This is achieved by projecting each word into n-dimensional space. Words with similar meanings occupy similar regions of this space; the closer two words are, the more similar they are. And often the vectors between words represent useful relationships, such as gender, verb tense, or even geopolitical relationships.

<img src="images/embedding-words.png" width="100%" align-center="true" alt="" title="Gated Recurrent Unit (GRU)" />

Training embeddings on a large dataset from scratch requires a huge amount of data and computation. So, instead of doing it ourselves, we'd normally use a pre-trained embeddings package such as [GloVe](https://nlp.stanford.edu/projects/glove/) or [word2vec](https://mubaris.com/2017/12/14/word2vec/). When used this way, embeddings are a form of transfer learning. However, since our dataset for this project has a small vocabulary and little syntactic variation, we'll use Keras to train the embeddings ourselves.

##### &nbsp;

### Encoder & Decoder

Our sequence-to-sequence model links two recurrent networks: an encoder and decoder. The encoder summarizes the input into a context variable, also called the state. This context is then decoded and the output sequence is generated.

##### &nbsp;

### Bidirectional Layer

The encoder only has historical context. But, providing future context can result in better model performance. This may seem counterintuitive to the way humans process language, since we only read in one direction. However, humans often require future context to interpret what is being said.

To implement this, we train two RNN layers simultaneously. The first layer is fed the input sequence as-is and the second is fed a reversed copy.

##### &nbsp;

### Hidden Layer &mdash; Gated Recurrent Unit (GRU)

Now let's make our RNN a little bit smarter. Instead of allowing _all_ of the information from the hidden state to flow through the network, what if we could be more selective? Perhaps some of the information is more relevant, while other information should be discarded. This is essentially what a gated recurrent unit (GRU) does.

There are two gates in a GRU: an update gate and reset gate. [This article](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) by Simeon Kostadinov, explains these in detail. To summarize, the **update gate (z)** helps the model determine how much information from previous time steps needs to be passed along to the future. Meanwhile, the **reset gate (r)** decides how much of the past information to forget.

##### &nbsp;

### Final Model

Now that we've discussed the various parts of our model, let's take a look at the code.

```python

def  model_final (input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Building and training a model that incorporates embedding, encoder-decoder, and bidirectional RNN
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Hyperparameters
    learning_rate = 0.003

    # Build the layers
    model = Sequential()
    # Embedding
    model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1],
                         input_shape=input_shape[1:]))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(output_sequence_length))

    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

    # Compiling model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
```

##### &nbsp;

## Results

The results from the final model can be found in cell 22 of the notebook.

Validation accuracy: 97.7%

Training time: 24 epochs

##### &nbsp;

### Contact

I hope you found this useful. If you have any feedback, I’d love to hear it.

If you’d like to inquire about collaboration or career opportunities you can find me [here on LinkedIn](https://www.linkedin.com/in/aditya-halder-007/).
