# A Python Program to implement Machine Learning for the Game Tic Tac Toe (3x3) using Reinforcement Learning (Q learning technique) and tensorflow. 
# 
#   Note: mistakes were made, especially spelling mistakes.
#   
#


# imports
import time
import tensorflow as tf
import random
import numpy as np
from pathlib import Path
import os
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Varibles 
game_rows = rows = 3
game_cols = cols = 3
winning_length = 3
boardSize = rows * cols
actions = rows * cols
won_games = 0
lost_games = 0
draw_games = 0
layer_1_w = 750
layer_2_w = 750
layer_3_w = 750


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


# greedy policy for selecting an action
# the higher the value of e the higher the probability of an action being random.
epsilon = 1.0

# Discount factor -- determines the importance of future rewards
GAMMA = 0.9


# swaps X's to O's and vice versa
def InverseBoard(board):
    temp_board = np.copy(board)
    rows, cols = temp_board.shape
    for r in range(rows):
        for c in range(cols):
            temp_board[r,c] *= -1
    return temp_board.reshape([-1])

# returns true if the game is completed for a given board
def isGameOver(board):
    temp = None
    rows , cols = board.shape

    ## ROWS
    for i in range(rows):
        temp = getRowSum(board, i)
        if checkValue(temp):
            return True
    ## COLS
    for i in range(cols):
        temp = getColSum(board, i)
        if checkValue(temp):
            return True

    ## Diagonals
    temp = getRightDig(board)
    if checkValue(temp):
        return True

    temp = getLeftDig(board)
    if checkValue(temp):
        return True

    # ## Does not contain empty places
    # empty_place_exist = False
    # for r in range(rows):
    #     for c in range(cols):
    #         if(board[r,c] == 0):
    #             empty_place_exist = True
    # if not empty_place_exist:
    #     return True

    return False

## support function
def getRowSum(board , r):
    rows , cols = board.shape
    sum = 0
    for c in range(cols):
        sum = sum + board[r,c]
    return sum

## support function
def getColSum(board , c):
    rows , cols = board.shape
    sum = 0
    for r in range(rows):
        sum = sum + board[r,c]
    return sum

## support function
def getLeftDig(board):
    rows , cols = board.shape
    sum = 0
    for i in range(rows):
        sum = sum + board[i,i]
    return sum

## support function
def getRightDig(board):
    rows , cols = board.shape
    sum = 0
    i = rows - 1
    j = 0
    while i >= 0:
        sum += board[i,j]
        i = i - 1
        j = j + 1
    return sum

## support function
def checkValue(sum):
    if sum == -3 or sum == 3:
        return True


# creates the network
def createNetwork():
    # network weights and biases

    W_layer1 = weight_variable([boardSize, layer_1_w])
    b_layer1 = bias_variable([layer_1_w])

    W_layer2 = weight_variable([layer_1_w, layer_2_w])
    b_layer2 = bias_variable([layer_2_w])

    W_layer3 = weight_variable([layer_2_w, layer_3_w])
    b_layer3 = bias_variable([layer_3_w])

    o_layer = weight_variable([layer_3_w, actions])
    o_bais  = bias_variable([actions])

    # input Layer
    x = tf.placeholder("float", [None, boardSize])

    # hidden layers
    h_layer1 = tf.nn.relu(tf.matmul(x,W_layer1) + b_layer1)
    h_layer2 = tf.nn.relu(tf.matmul(h_layer1,W_layer2) + b_layer2)
    h_layer3 = tf.nn.relu(tf.matmul(h_layer2,W_layer3) + b_layer3)

    # output layer
    y = tf.matmul(h_layer3,o_layer) + o_bais
    prediction = tf.argmax(y[0])

    return x,y, prediction


def tainNetwork():
    print()

    # create network
    inputState , Qoutputs, prediction = createNetwork()

    # calculate the loss
    targetQOutputs = tf.placeholder("float",[None,actions])
    loss =  tf.reduce_mean(tf.square(tf.subtract(targetQOutputs, Qoutputs)))

    # train the model to minimise the loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # creating a sesion
    sess = tf.InteractiveSession()

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # load a saved model
    step = 0
    iterations = 0

    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = saver.restore(sess,checkpoint.model_checkpoint_path)
        print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
    else:
        print("Could not find old network weights")
    iterations += step

    print(time.ctime())

    ## define maximum number of matches for inital interation
    tot_matches = 60000
    number_of_matches_each_episode = 500
    max_iterations = tot_matches / number_of_matches_each_episode
    
    # defines the rate at which epsilon should decrease
    e_downrate = 0.9 / max_iterations

    print("e down rate is ",e_downrate)

    # initalise e with inital epsilon value here.
    e = epsilon

    print("max iteration = {}".format(max_iterations))
    print()
    
    run_time = 0
    while "ticky" != "tacky":
        sys.stdout.flush()
        start_time = time.time()
        episodes = number_of_matches_each_episode
        global won_games
        global lost_games
        global draw_games

        # sum of the losses while training the model.
        total_loss = 0

        epchos = 100
        GamesList = []

        for i in range(episodes):
            completeGame, victory = playaGame(e,sess,inputState, prediction,Qoutputs)
            GamesList.append(completeGame)
            

        for k in range(epchos):
            random.shuffle(GamesList)
            for i in GamesList:
                len_complete_game = len(i)
                loop_in = 0
                game_reward = 0
                while loop_in < len_complete_game:
                    j = i.pop()
                    currentState = j[0]
                    action = j[1][0]
                    reward = j[2][0]
                    nextState = j[3]

                    ## Game end reward
                    if loop_in == 0:
                        game_reward = reward
                    else:
                        #obtain q values for next sate using the network
                        nextQ = sess.run(Qoutputs,feed_dict={inputState:[nextState]})
                        maxNextQ = np.max(nextQ)
                        game_reward = GAMMA * ( maxNextQ )

                    
                    targetQ = sess.run(Qoutputs,feed_dict={inputState:[currentState]})

                    # once we calculate the reward to the paticular action, we must also add the -1 reward for all the illegal moves in the q value
                    # one might say this is cheating , but it speeds up the process.
                    for index,item in enumerate(currentState):
                        if item != 0:
                            targetQ[0,index] = -1

                    targetQ[0,action] = game_reward

                    loop_in += 1
                    t_loss = 0
                    #Train our network using the targetQ

                    
                    t_loss=sess.run([train_step,Qoutputs,loss],feed_dict={inputState:[currentState], targetQOutputs:targetQ})
                    total_loss += t_loss[2]

        iterations += 1
        time_diff = time.time()-start_time
        run_time += time_diff
        print("iteration {} completed with {} wins, {} losses {} draws, out of {} games played, e is {} \ncost is {} , current_time is {}, time taken is {} , total time = {} hours \n".format(iterations,
        won_games,lost_games,draw_games,episodes,e*100,total_loss,time.ctime(),time_diff,(run_time)/3600))
        start_time = time.time()
        total_loss = 0
        won_games = 0
        lost_games = 0
        draw_games = 0

        # decrease e value slowly.
        if e > -0.2:
            e -= e_downrate
        else:
             e = random.choice([0.1,0.05,0.06,0.07,0.15,0.03,0.20,0.25,0.5,0.4])

        #print(wins,loss,(episodes-wins-loss))
        saver.save(sess, "./model/model.ckpt",global_step=iterations)




# plays a game and returns a list with all states, actions and final reward.
def playaGame(e,sess,inputState, prediction, Qoutputs):
    global won_games
    global lost_games
    global draw_games

    win_reward = 10
    loss_reward = -1
    draw_reward = 3

    ## create the entire game memory object that contains the memories for the game
    ## and an empty board
    completeGameMemory = []
    myList = np.array([0]*(rows*cols)).reshape(3,3)

    ## randomly chose a turn 1 is ours -1 is oppnents
    turn = random.choice([1,-1])

    ## if opponents turn let him play and set the inital state
    if(turn == -1):
        initial_index = random.choice(range(9))
        best_index, _= sess.run([prediction,Qoutputs], feed_dict={inputState : [np.array(np.copy(myList).reshape(-1))]})
        initial_index = random.choice([best_index,initial_index,best_index])
        myList[int(initial_index/3),initial_index%3] = -1
        turn = turn * -1

    ## while the game is not over repat, our move then opponents move
    while(True):

        ## create a memory which will hold the current inital state, the action thats taken, the reward the was recieved, the next state
        memory = []

        ## create a copy of the board which is linear
        temp_copy = np.array(np.copy(myList).reshape(-1))

        ## fetch all the indexes that are free or zero so those can used for playing next move
        zero_indexes = []
        for index,item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

        ## if no index is found which is free to place a move exit as the game completed with slight reward. better to draw then to lose right ?
        if len(zero_indexes) == 0:
            reward = draw_reward
            completeGameMemory[-1][2][0] = reward
            draw_games += 1
            break

        ## if free indexs are found randomly select one which will be later can be used as the action.
        selectedRandomIndex = random.choice(zero_indexes)

        ## calculate the prediction from the network which can be later used as an action with some probability
        pred, _ = sess.run([prediction,Qoutputs], feed_dict={inputState : [temp_copy]})

        ## since the netowrk can be messy and inacurate check if the prediction is correct first.
        isFalsePrediction = False if temp_copy[pred] == 0 else True

        ## lets add the inital state to the current memory
        memory.append(np.copy(myList).reshape(-1))

        ## Lets pick an action with some probability, exploration and exploitation
        if random.random() > e: #and isFalsePrediction == False: #expliotation
            action = pred
        else: # exploration, explore with valid moves to save time.
            random_action = random.choice(range(9))
            action = selectedRandomIndex
            #action = random.choice([selectedRandomIndex,random_action])
            #action = random.choice(range(9))

        ## lets add the action to the memory
        memory.append([action])

        ## randomly plays a wrong move.. unlucky, however.
        if action not in zero_indexes:
            reward = loss_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            lost_games +=1
            break

        ## update the board with the action taken
        myList[int(action/game_rows),action%game_cols] = 1

        ## now calcualte the reward.
        reward = 0

        ## if we choose an action thats invalid, boo we get no reward and opponent wins
        if isFalsePrediction == True and action == pred:
            reward = loss_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            lost_games +=1
            break

        ## if after playing our move the game is completed then yay we deserve a reward and its the final state
        if(isGameOver(myList)):
            reward = win_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            won_games +=1
            break



        # Now lets make a move for the opponent

        ## same as before, but since we are finding a move for the opponent we use the inverse board
        ## to calculate the prediction
        temp_copy_inverse = np.array(np.copy(InverseBoard(myList)).reshape(-1))
        temp_copy = np.array(np.copy(myList).reshape(-1))
        zero_indexes = []
        for index,item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

        ## if opponent has no moves left that means that the last move was the final move and its a draw so some reward
        if len(zero_indexes) == 0:
            reward = draw_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            draw_games+=1
            break

        ## almost same as before
        selectedRandomIndex = random.choice(zero_indexes)
        pred, _ = sess.run([prediction,Qoutputs], feed_dict={inputState : [temp_copy_inverse]})
        isFalsePrediction = False if temp_copy[pred] == 0 else True

        ## we want opponet to play good sometimes and play bad sometimes so 33.33% ish probability
        action = None

        if(isFalsePrediction == True):
            action = random.choice([selectedRandomIndex])
        else:
            action = random.choice([selectedRandomIndex,pred,pred,pred,pred])
            #action = random.choice([selectedRandomIndex,pred])
        # if e < 0.4 and isFalsePrediction == False:
        #     action = pred

        # testing
        temp_copy2 = np.copy(myList).reshape(-1)
        if temp_copy2[action] != 0:
            print("big time error here ",temp_copy2 , action)
            return

        ## update the board with opponents move
        myList[int(action/game_rows),action%game_cols] = -1

        ## if after opponents move the game is done meaning opponent won, boo..
        if isGameOver(myList) == True:
            reward = loss_reward
            memory.append([reward])
            #final state
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            lost_games +=1
            break

        ## if no one won and game isn't done yet then lets continue the game
        memory.append([0])
        memory.append(np.copy(myList.reshape(-1)))

        ## lets add this move to the complete game memory
        completeGameMemory.append(memory)

    # return the complete game memory and the last set reward
    return completeGameMemory,reward


# not used.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass



if __name__ == "__main__":
    #sys.stdout = Logger()
    tainNetwork()
