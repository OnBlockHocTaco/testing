# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        """
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
        """

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #for score in scores:
        #    print("Score",score)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        #print("Picked Action",legalMoves[chosenIndex])
        return legalMoves[chosenIndex]





    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #legalMoves = successorGameState.getLegalActions()
        allFood = newFood.asList()
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Determine target

        foodList = newFood.asList()

        # Nerf stop

        # score = 0

        # Deter proximity to ghost
        if newPos in successorGameState.getGhostPositions():
            return 0

        # Boost if newPos is in currFoods
        if newPos in currentGameState.getFood().asList():
            return 1.1

        # Get farthest food
        farthestFood = foodList[0]
        minDistance = manhattanDistance(newPos, farthestFood)
        for food in foodList:
            currDistance = manhattanDistance(newPos, food)
            if currDistance < minDistance:
                minDistance = currDistance
                farthestFood = food

        return 1 / minDistance





def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        pacmanActions = gameState.getLegalActions(0)
        toPerform = None
        maxVal = float("-inf")
        for action in pacmanActions:
            if gameState.getNumAgents() == 1:
                val = self.getMiniMaxValue(gameState.generateSuccessor(0, action), self.depth - 1, 0)
            else:
                val = self.getMiniMaxValue(gameState.generateSuccessor(0, action), self.depth, 1)
            if val > maxVal:
                maxVal = val
                toPerform = action
        return toPerform


    def getMiniMaxValue(self, gameState: GameState, depth: int, agent: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        elif agent == 0:
            maxVal = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                if gameState.getNumAgents() == 1:
                    val = self.getMiniMaxValue(gameState.generateSuccessor(0, action), depth - 1, 0)
                else:
                    val = self.getMiniMaxValue(gameState.generateSuccessor(0, action), depth, agent + 1)
                if val > maxVal:
                    maxVal = val
            return maxVal
        elif agent == gameState.getNumAgents() - 1:
            minVal = float("inf")
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val = self.getMiniMaxValue(gameState.generateSuccessor(agent, action), depth - 1, 0)
                if val < minVal:
                    minVal = val
            return minVal
        else:
            minVal = float("inf")
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val = self.getMiniMaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1)
                if val < minVal:
                    minVal = val
            return minVal



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanActions = gameState.getLegalActions(0)
        toPerform = None
        maxVal = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in pacmanActions:
            newGameState = gameState.generateSuccessor(0, action)
            if gameState.getNumAgents() == 1:
                val, a, b = self.getAlphaBetaValue(newGameState, self.depth - 1, 0, alpha, beta)
            else:
                val, a, b = self.getAlphaBetaValue(newGameState, self.depth, 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                toPerform = action
            alpha = max(a, maxVal)
        return toPerform

    def getAlphaBetaValue(self, gameState: GameState, depth: int, agent: int, alpha: int, beta: int):
        print("Alpha is " + str(alpha))
        print("Beta is " + str(beta))
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), alpha, beta]
        elif agent == 0:
            maxVal = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                if gameState.getNumAgents() == 1:
                    val, a, b = self.getAlphaBetaValue(gameState.generateSuccessor(0, action), depth - 1, 0, alpha, beta)
                else:
                    val, a, b = self.getAlphaBetaValue(gameState.generateSuccessor(0, action), depth, agent + 1, alpha, beta)
                maxVal = max(maxVal, val)
                if maxVal > b:
                    return [maxVal, a, b]
                alpha = max(a, maxVal)
            return [maxVal, alpha, beta]
        elif agent == gameState.getNumAgents() - 1:
            minVal = float("inf")
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val, a, b = self.getAlphaBetaValue(gameState.generateSuccessor(agent, action), depth - 1, 0, alpha, beta)
                minVal = min(minVal, val)
                if minVal < a:
                    return [minVal, alpha, beta]
                beta = min(b, minVal)
            return [minVal, alpha, beta]
        else:
            minVal = float("inf")
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val, a, b = self.getAlphaBetaValue(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                minVal = min(minVal, val)
                if minVal < a:
                    return [minVal, alpha, beta]
                beta = min(b, minVal)
            return [minVal, alpha, beta]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacmanActions = gameState.getLegalActions(0)
        toPerform = None
        maxVal = float("-inf")
        for action in pacmanActions:
            if gameState.getNumAgents() == 1:
                val = self.getExpectimaxValue(gameState.generateSuccessor(0, action), self.depth - 1, 0)
            else:
                val = self.getExpectimaxValue(gameState.generateSuccessor(0, action), self.depth, 1)
            if val > maxVal:
                maxVal = val
                toPerform = action
        return toPerform


    def getExpectimaxValue(self, gameState: GameState, depth: int, agent: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        elif agent == 0:
            maxVal = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                if gameState.getNumAgents() == 1:
                    val = self.getExpectimaxValue(gameState.generateSuccessor(0, action), depth - 1, 0)
                else:
                    val = self.getExpectimaxValue(gameState.generateSuccessor(0, action), depth, agent + 1)
                if val > maxVal:
                    maxVal = val
            return maxVal
        elif agent == gameState.getNumAgents() - 1:
            #minVal = float("inf")
            total = 0
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val = self.getExpectimaxValue(gameState.generateSuccessor(agent, action), depth - 1, 0)
                total += val
            return total / len(legalActions)
        else:
            #minVal = float("inf")
            total = 0
            legalActions = gameState.getLegalActions(agent)
            for action in legalActions:
                val = self.getExpectimaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1)
                total += val
            return total / len(legalActions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    cost = 0
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    #newGhostStates = successorGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # Determine target
    foodList = food.asList()
    # Deter proximity to ghost
    if pos in currentGameState.getGhostPositions():
        cost = float("inf")
    # Boost if newPos is in currFoods
    if pos in foodList:
        cost += 1.1
    # Get farthest food
    farthestFood = foodList[0]
    minDistance = manhattanDistance(pos, farthestFood)
    for food in foodList:
        currDistance = manhattanDistance(pos, food)
        if currDistance < minDistance:
            minDistance = currDistance

    return 1 / (cost + minDistance)




# Abbreviation
better = betterEvaluationFunction
