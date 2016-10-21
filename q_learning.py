#from pongv1 import WINDOWWIDTH, WINDOWHEIGHT, LINETHICKNESS, PADDLESIZE, PADDLEOFFSET, SCOREWINNER, WINNER, ballDirX, ballX, ballY, playerTwoPosition, playerOneLastPosition, playerTwoLastPosition, ballDistanceY 
import pandas as pd
import random
from ggplot import *


class QLearn:

    #q-learning params
    __alpha = 0     #learning rate, how important new discoveries are
    __gamma = 0     #discount rate, how important past decisions are
    __epoch = 0     #number of cycles the algorithm has been learning
    __max_epoch = 0 #max number of cycles
    __epsilon = 0   #greedy choice param
    __model = None
    __last_action = ''
    __current_action = ''
    __current_state = '' 
    __last_state = '' 

    __RESULT_X_EPOCH_DATAFRAME = None
    __POINTS_X_EPOCH_DATAFRAME = None
    


    #sets default values for alpha and gamma params
    def __init__(self, model,**kwargs):
        self.__alpha = kwargs.get('alpha',0.2)
        self.__gamma = kwargs.get('gamma',0.6)
        self.__epsilon = kwargs.get('epsilon',0.8)
        self.__epoch = kwargs.get('epoch',1)
        self.__model = model

        try:
            self.__RESULT_X_EPOCH_DATAFRAME = pd.read_csv('results.csv',index_col=0)
            self.__epoch = (self.__RESULT_X_EPOCH_DATAFRAME.loc[self.__RESULT_X_EPOCH_DATAFRAME['Epoch'].idxmax()]['Epoch']) + 1
        except:
            self.__RESULT_X_EPOCH_DATAFRAME = pd.DataFrame(columns=['Epoch','Result'])

        try:
            self.__POINTS_X_EPOCH_DATAFRAME = pd.read_csv('points.csv',index_col=0)
        except:
            self.__POINTS_X_EPOCH_DATAFRAME = pd.DataFrame(columns=['Epoch','Score'])


    #this is the main learn function called from the game script in every iteration
    def learn(self):
        if self.__last_action != '' and self.__last_state != '':
            policy_dataframe = self.__model.getPOLICY_DATAFRAME()
            reward_dataframe = self.__model.getREWARD_DATAFRAME()

            learned_value = reward_dataframe.loc[self.__last_state][self.__last_action[0]]
            reward = False
            if learned_value == 1:
                reward = True
                print('Got rewarded!')
            learned_value += self.__gamma * policy_dataframe.loc[self.__current_state].max(skipna=False)

            #print('learned_value =  '+str(learned_value))
            old_value = policy_dataframe.loc[self.__last_state][self.__last_action[0]]

            #print('old_value =  ' + str(old_value))
            new_value = (1 - self.__alpha) * old_value + self.__alpha * learned_value

            if reward:
                print('New Q value =  ' + str(new_value))
            policy_dataframe.set_value(self.__last_state,self.__last_action[0],new_value)

            #self.__model.savePolicy()

    #returns one of the 3 possible moves: UP, DOWN or STOP and the Q value in a list
    def next_move(self,ballX,ballY,paddle1,paddle2,ballXspeed,ballYspeed):
        #loads csv file if not loaded.
        if not self.__model.getPOLICY_DATAFRAME().empty:       
            game_state = self.__model.getGameState(ballX,ballY,paddle1,paddle2,ballXspeed,ballYspeed)
            
            next_options = self.__model.getPOLICY_DATAFRAME().loc[game_state]
            max_value = next_options.loc[next_options.idxmax(skipna=False)]
            if max_value >= self.__epsilon:
                result = [next_options.idxmax(skipna=False),next_options.max(skipna=False)]               
            else:
                result = [random.choice(self.__model.getGAME_ACTIONS()),0]
            self.__last_action = self.__current_action
            self.__last_state =self.__current_state
            self.__current_action = result
            self.__current_state = game_state

            #print('Next action: '+self.__current_action[0])
            return result



    #checks is present state generated a reward
    def getReward(self,ballX,ballY,paddle1,paddle2,ballXspeed,ballYspeed,action):
        if not self.__model.getREWARD_DATAFRAME().empty:
            game_state = self.__model.getGameState(ballX,ballY,paddle1,paddle2,ballXspeed,ballYspeed)
            rewards = self.__model.getREWARD_DATAFRAME().loc[game_state]

            return rewards[action]

    def incrementEpoch(self,score,result):
        self.__RESULT_X_EPOCH_DATAFRAME = self.__RESULT_X_EPOCH_DATAFRAME.append(pd.DataFrame([[self.__epoch,result]],columns=['Epoch','Result']),ignore_index=True)
        self.__RESULT_X_EPOCH_DATAFRAME.to_csv('results.csv')
        self.__POINTS_X_EPOCH_DATAFRAME = self.__POINTS_X_EPOCH_DATAFRAME.append(pd.DataFrame([[self.__epoch,score]],columns=['Epoch','Points']),ignore_index=True)
        self.__POINTS_X_EPOCH_DATAFRAME.to_csv('points.csv')
        self.__epoch += 1

    def plot_data(self):
        q_plot = ggplot(aes(x='Epoch',y='Points'),data=self.__POINTS_X_EPOCH_DATAFRAME.reset_index()) + geom_line()
        ggsave('points.png',q_plot)


    ##Getters and Setters
    def getCurrentState(self):
        return self.__current_state

    
