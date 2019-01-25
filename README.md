# sentiment_analysis
A beginner's approach for [this] (https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) Kaggle problem.

## File Structure
processKaggleDataset.py: Reads and cleans the data. Also provides the word2vec matrix for the dataset.
trainCNN.py: Contains the code to create and train the network with Keras.
output.csv: Contains the predicted labels for the phrases in the test file in the form
 (PhraseId, Sentiment)

## Data Cleaning and preprocessing
The following python functions and statements are in file processKaggleDataset.py.
revs, vocab = build_data_train_test(data_train, data_test, train_ratio=0.8, clean_string=True)
This function takes the data and splits them into training and validation set. Also it cleans all the data including the test data and returns the dictionary ‘vocab’ and the list ‘revs’. ‘Vocab’ is a dict which contains all the words in the dataset along with the number of phrases that they appeared in. ‘revs’ is a list containing the phrases along with their sentiment label.
w2v = load_bin_vec(w2v_file, vocab)
This function uses the Google word2vec for word embedding. It returns a dict containing the word vectors for all the words in the ‘vocab’.
W, word_idx_map = get_W(w2v)
This function returns the matrix W where W[i] is the vector for word indexed by i. The word_idx_map is a dict mapping word with their indices.

## CNN Model:
Our Convolutional Neural Network has the following structure(layers):
1.	Embedding 
2.	Convolution 
3.	Activation(Relu)
4.	MaxPooling 
5.	Fully connected NN output layer
6.	Activation(Softmax)
The CNN is trained with the following hyperparameters:
Number of feature maps (filters) = 300
Kernel size of convolution = 8
Lossfunction = categorical_crossentropy
Optimizer = Adadelta

## Training and Result
The CNN is trained for 3 epochs since it takes 4-5 hours to run one epoch. The validation accuracy after 3 epochs is 0.66.
If we train the network for 10-15 epochs, the accuracy will certainly increase since it will take at least 10-15 epochs to train the model for 156000 training examples.
Another point to keep in mind is that the top team at the leaderboard for this competition has accuracy of 0.75 with only 2 teams having score above 0.7. So any score around 0.65 is not that bad considering just 3 epochs.
