wyEnv.py: create the basic game environment for agents to take actions.

wyAgent.py: create DQN-Agent for the game.

wyPolicy: E-Greedy policy implementation.

wyGame.py: construct the game environment and offer it to agents to learn

wyCritic.h5: regularly saved checkpoint, accuracy 0.9716

wyconReward: rewards track of training of wyCritic.h5.

wyconEncoder.py: CNN encoder for MNIST.

modelEncoder.h5: saved model for wyconEncoder.

-------------------------------------------------------------------------------------------------------------------
For training:

enter in command line: python wyGame.py

For Testing:

change the line215 in wyGame.py to: g.play(2000, False)
