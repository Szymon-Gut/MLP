# Repository contatins MLP implementation from scratch


### Goal 

Repository contains full implementation of multilayer perceptron. In subsequent milestones carried out as part of the Computational Intelligence course, I had to extend the implementation of the perceptron with new features, such as new activation functions while ensuring correct operation of backpropagation for each of them, regularization methods like L1 and L2, implementing the early stopping mechanism, adding optimization in the form of momentum learning and RMSprop, and many others.

### Project structure

- **data/**: Contains the data used to test our solutions. To successfully pass each milestone, our neural network implementation with additional features had to achieve a sufficiently low threshold of loss function.

- **notebooks/**: Contains 3 notebooks presenting the performance of my implementation at each milestone. These notebooks showcase learning trajectories and other characteristics defining training, allowing the course instructor to see the threshold of loss our implementation was able to reach (which was an important evaluation criterion).

- **network.py**: Contains the full implementation of the neural network class.

- **layer.py**: Contains the implementation of the perceptron layer class used in building the network and imported into the neural network class.

- **activation_functions.py**: Contains the implementation of activation functions used by me, along with calculating their derivatives.

- **metrics.py**: Contains the implementation of metrics used by me, tailored to my implementation of the neural network. There are very few metrics present because only those were needed to document the expected results by the Professor. However, it's very easy to expand this with new metrics.

- **prepare_data.py**: Contains a script used to prepare datasets for loading in the testing process of the implementation provided in each milestone.

Finally, the most important file is **MLP_sprawozdanie.pdf**, which is a report summarizing the laboratories related to the implementation of MLP from scratch. It extensively describes the results and outcomes of each milestone, along with a detailed description of what was implemented in them (unfortunately only in Polish).


### Usage

The usage is thoroughly described in the code documentation, which has been provided, as well as in the notebooks available in the notebooks folder.


*Project was realised during Methods of Computional Inteligence classes at Warsaw University Of Technology *
