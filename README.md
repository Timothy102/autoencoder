# Autoencoder

Autoencoders are essentially neural networks designed for learning representations of data also called the codings in an unsupervised manner. These codings usually have much lower dimensionality than the input data, which makes autoencoder a great dimensionality reduction solution. 

For example, the notorious PCA (Principal Component Analysis) attempts to come up with a new lower dimensional hyperplane via the covariance matrix and the corresponding eigenvectors, whereas autoencoders never attempted to solve a linear problem, therefore they're capable of doing a nonlinear dimensionality reduction.

As you might have guessed, it's great because it uses an encoder on the language you' re trying to translate and does so by creating an inner representation or data codings. A decoder then unveils these codings into a desired output.

<img src="https://nathanhubens.github.io/posts/images/autoencoders/AE.png" alt="drawing" width="750"/>

