# 2026-07-09 — network.py complete
- Built: Network.py importing the layers.py file. Made predict, loss, gradient, and accruacy methods
- Description: The predict method allows the neural network to predict what the hand written number is. The loss method calculates how bad the neural network guessed. The gradient method stores just how much each weight and bias effected the bad results. And the accuracy method tells us how accurate the network is. 
- Still confused by: I am still getting alot of help by Claude. I ask questions and Claude guides me through. I am still learning so much and it is hard to grasp the full picture or the fine details.

# 2026-07-10 - train.py completed; Updated the mnist experiment to easily allow shifts in the number of nodes in each layer as well the amount of layers in total; save/load
- Built: train.py importing from layers, network, and dataset.mnist
- Description, trained the mnist layer 100,000 times brut forcing my way to 97.99 accuracy
- Still confused by: Still, much of what I am doing is just following orders from Claude. I don't fully understand the code and how it operates. I want to though. 
- Implemented the He-init, allowing for proper training of deeper neural networks. 
- Achieved 99.19 accuracy with [784,1000,1000,10] neural network layers
