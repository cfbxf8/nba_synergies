# nba_synergies

Get NBA Player Capabilities, synergies between NBA players, and predictions for future matchups using a Network Graph representation.

The current way that team-output is forecasted, simply as the sum of the individual capabilities is not often seen in reality.
There has been little work in modeling the capabilities of teams other than through this lense.
This new method, originally suggested by [Carnegie Mellon robotics researchers](https://github.com/cfbxf8/nba_synergies/tree/master/research%20papers), addresses this problem by providing an innovative way to model team synergies, in addition to some compelling benefits.

![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.001.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.002.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.003.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.004.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.005.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.006.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.007.jpeg)
```
Because this is a new and relatively unknown methodology, to fit them model I had to build the unweighted and weighted Synergy Graphs Classes from scratch including the Simulated Annealing and Genetic Algorithm minimization functions.
Representing this connected weighted graph distance as a matrix, 30x improvement in computation time for convergence.
```
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.008.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.009.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.010.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.011.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.012.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.013.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.014.jpeg)
![alt text](https://github.com/cfbxf8/nba_synergies/blob/master/imgs/Presentation.015.jpeg)

Special thanks again to Somchaya Liemhetcharat, Manuela Veloso, and Yicheng Luo for their research in this area!
