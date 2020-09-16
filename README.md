# Sentiment-with-Deep-Neural-Networks
We implemented Logistic regression and Naive Bayes for sentiment analysis. However if we give the old models an example like:

<center> <span style='color:blue'> <b>This movie was almost good.</b> </span> </center>

The model would have predicted a positive sentiment for that review. However, that sentence has a negative sentiment and indicates that the movie was not good. To solve those kinds of misclassifications, we will write a program that uses deep neural networks to identify sentiment in text. Concretely, we will: 

- Understand how we can build/design a model using layers
- Train a model using a training loop
- Use a binary cross-entropy loss function
- Compute the accuracy of your model
- Predict using your own input

- Indeed most of the deep nets we will be implementing will have a similar structure. The only thing that changes is the model architecture, the inputs, and the outputs. Before starting, we will introduce you to the Google library `trax` that we use for building and training models.
 

- Trax source code can be found on Github: [Trax](https://github.com/google/trax)
- The Trax code also uses the JAX library: [JAX](https://jax.readthedocs.io/en/latest/index.html)
