import numpy
import pandas as pd

class LocationModel:

    #imported params
    __WINDOWWIDTH = 0
    __WINDOWHEIGHT  = 0
    __LINETHICKNESS = 0
    __PADDLESIZE = 0
    __PADDLEOFFSET = 0
    __SCOREWINNER = 0
    __ENEMY_PADDLE_STATES = 0   
    __PLAYER_PADDLE_STATES = 0  
    __ballStateCoordinates = []   #array with all the coordinates for ball states
    __ballStateNames = []     #array with all ball state names
    __playerPaddleCoordinates = []    #array with edges for player paddle states
    __playerPaddleNames = []  #array with all player state names
    __enemyPaddleCoordinates = []  #array with edges for enemy paddle states
    __enemyPaddleNames = []   #array with enemy paddle state names
    __GAME_STATES = []    #list of all game states.
    __GAME_ACTIONS = ['UP','DOWN','STOP']     #list of possible actions for the player
    __REWARD_STATES = []        #list of all state names with reward
    __LAST_STATE = None
    __REWARD_CSV_PATH = ''
    __POLICY_CSV_PATH = '' 
    __REWARD_DATAFRAME = None
    __POLICY_DATAFRAME = None

    def __init__(self, WINDOWWIDTH, WINDOWHEIGHT, LINETHICKNESS, PADDLEOFFSET,PADDLESIZE,SCOREWINNER):

        self.__WINDOWHEIGHT = WINDOWHEIGHT
        self.__WINDOWWIDTH = WINDOWWIDTH
        self.__LINETHICKNESS = LINETHICKNESS
        self.__PADDLEOFFSET = PADDLEOFFSET
        self.__PADDLESIZE = PADDLESIZE
        self.__SCOREWINNER = SCOREWINNER

        self.__ENEMY_PADDLE_STATES = 3   
        self.__PLAYER_PADDLE_STATES = 20   

        self.__ballStateCoordinates.append([0,PADDLEOFFSET + LINETHICKNESS ,0,WINDOWHEIGHT]) #state 0 = ignore
        self.__ballStateCoordinates.append([(WINDOWWIDTH/2) + 1 ,WINDOWWIDTH ,0,WINDOWHEIGHT]) #state 1 = ignore

        for i in range(41):
            self.__ballStateNames.append('B_'+str(i))
        #print(str(self.ballStateNames))  #this is a debug print

        #first ball state row
        for rect_line in range(20):
            self.__ballStateCoordinates.append([29,50,rect_line*24,(rect_line+1)*24])
        #second ball state row
        for rect_line in range(10):
            self.__ballStateCoordinates.append([51,100,rect_line*48,(rect_line+1)*48])
        #third ball state row
        for rect_line in range(6):
            self.__ballStateCoordinates.append([101,180,rect_line*80,(rect_line+1)*80])
        #fourth ball state row
        for rect_line in range(3):
            self.__ballStateCoordinates.append([181,320,rect_line*160,(rect_line+1)*160])
        #print(str(self.__ballStateCoordinates))  #this is a debug print


        for player_paddle_area_id in range(0,self.__PLAYER_PADDLE_STATES):
            self.__playerPaddleCoordinates.append([0,PADDLEOFFSET + LINETHICKNESS,(self.__PADDLESIZE/2 + player_paddle_area_id * ((WINDOWHEIGHT-PADDLESIZE)/self.__PLAYER_PADDLE_STATES))+1,self.__PADDLESIZE/2 + (player_paddle_area_id + 1)* ((WINDOWHEIGHT-PADDLESIZE)/self.__PLAYER_PADDLE_STATES) ])
            self.__playerPaddleNames.append('P1_'+str(player_paddle_area_id))
        #print(str(playerPaddleNames))  #this is a debug print


        for enemy_paddle_area_id in range(0,self.__ENEMY_PADDLE_STATES):
            self.__enemyPaddleCoordinates.append([WINDOWWIDTH - (PADDLEOFFSET + LINETHICKNESS),WINDOWWIDTH,(PADDLESIZE/2 + enemy_paddle_area_id * ((WINDOWHEIGHT-PADDLESIZE)/self.__ENEMY_PADDLE_STATES))+1,self.__PADDLESIZE/2 + (enemy_paddle_area_id + 1)* ((WINDOWHEIGHT-PADDLESIZE)/self.__ENEMY_PADDLE_STATES) ])
            self.__enemyPaddleNames.append('P2_'+str(enemy_paddle_area_id))
        #print(str(self.__enemyPaddleCoordinates))  #this is a debug print

        for ball_state in self.__ballStateNames:
            for player_state in self.__playerPaddleNames:
                for enemy_state in self.__enemyPaddleNames:
                    self.__GAME_STATES.append(ball_state+'_'+player_state+'_'+enemy_state)
        #print(str(GAME_STATES))  #this is a debug print

        for i in range(20):
            for enemy_state in self.__enemyPaddleNames:
                self.__REWARD_STATES.append(self.__ballStateNames[i + 2]+'_'+self.__playerPaddleNames[i]+'_'+enemy_state)       #states 0 and 1 are being ignored!
                """if i > 1:
                    self.__REWARD_STATES.append(self.__ballStateNames[i + 2]+'_'+self.__playerPaddleNames[i-1]+'_'+enemy_state)
                elif i < 18:
                    self.__REWARD_STATES.append(self.__ballStateNames[i + 2]+'_'+self.__playerPaddleNames[i+1]+'_'+enemy_state) """
        #self.__REWARD_STATES.append('B_2_P1_1_P2_0')
        #self.__REWARD_STATES.append('B_2_P1_1_P2_1')
        #self.__REWARD_STATES.append('B_2_P1_1_P2_2')
        #self.__REWARD_STATES.append('B_19_P1_18_P2_0')
        #self.__REWARD_STATES.append('B_19_P1_18_P2_1')
        #self.__REWARD_STATES.append('B_19_P1_18_P2_2')
        #print(str(self.__REWARD_STATES))    #this is a debug test

        #sets csv locations
        self.__POLICY_CSV_PATH = 'q_learning_policy_' + self.getName() + str(self.getPLAYER_PADDLE_STATES()) + '_' + str(self.getENEMY_PADDLE_STATES()) + '.csv'
        self.__REWARD_CSV_PATH = 'q_learning_reward_' + self.getName() + str(self.getPLAYER_PADDLE_STATES()) + '_' + str(self.getENEMY_PADDLE_STATES()) + '.csv'

        #loads dataframes
        self.loadRewardAndPolicy()
        self.saveReward()
        self.savePolicy()


    def getGameStateList(self):
        return self.__GAME_STATES


    #returns in which state the ball is, considering the division of the defenders field in N sub regions
    def getBallState(self,ballX, ballY):
        #print(str(self.__ballStateCoordinates))    #this is a debugging print
        for rectangle_id,area in enumerate(self.__ballStateCoordinates):
            
            if self.__isInsideRectangle(ballX + (self.__LINETHICKNESS / 2),ballY+ (self.__LINETHICKNESS / 2),*area):
                return 'B_'+str(rectangle_id)
        return 'error out of bounds!'

    def __isInsideRectangle(self,x,y,xLeft,xRight,yUp,yDown):
        if x>=xLeft and x<=xRight and y>=yUp and y<=yDown:
            return True
        else:
            return False

    #returns the player's paddle states, given the number of possible states.
    def getPlayerPaddleState(self,paddle1):
        #print(str(playerPaddleCoordinates))    #this is a debugging print
        for player_paddle_area_id, area in enumerate(self.__playerPaddleCoordinates):
            if  self.__isInsideRectangle((self.__PADDLEOFFSET+ self.__LINETHICKNESS / 2), paddle1 + (self.__PADDLESIZE / 2) ,*area ):
                return 'P1_'+str(player_paddle_area_id)
        print('error out of bounds in getPlayerPaddleState!')
        return 'error out of bounds!'

    #returns the enemy's paddle state, given the number of possible states.
    def getEnemyPaddleState(self,paddle2):
        #print(str(enemyPaddleCoordinates))    #this is a debugging print
        for enemy_paddle_area_id, area in enumerate(self.__enemyPaddleCoordinates):
            if self.__isInsideRectangle(self.__WINDOWWIDTH - self.__PADDLEOFFSET - (self.__LINETHICKNESS / 2), paddle2 + (self.__PADDLESIZE / 2) ,*area ):
                return 'P2_'+str(enemy_paddle_area_id)
        print('error out of bounds in getEnemyPaddleState!')
        return 'error out of bounds!'

    def getGameState(self,ballX,ballY,paddle1,paddle2):
        return self.getBallState(ballX,ballY) + '_' + self.getPlayerPaddleState(paddle1) + '_' + self.getEnemyPaddleState(paddle2)

    def getName(self):
        return 'location_model'

    def getNewRewardMatrix(self):

        state_action_values = numpy.zeros(shape=(len(self.__GAME_STATES),len(self.__GAME_ACTIONS)))      #creates a states x actions matrix with zeroes
        reward_dataframe = pd.DataFrame(state_action_values,index = self.__GAME_STATES,columns=self.__GAME_ACTIONS)
        for reward_state in self.__REWARD_STATES:
            reward_dataframe.set_value(reward_state,'STOP',1)
            reward_dataframe.set_value(reward_state,'UP',1)
            reward_dataframe.set_value(reward_state,'DOWN',1)
        #print(reward_dataframe.head(10))       #test print
        return reward_dataframe


    def getNewPolicyMatrix(self):

        state_action_values = numpy.zeros(shape=(len(self.__GAME_STATES),len(self.__GAME_ACTIONS)))      #creates a states x actions matrix with zeroes
        return pd.DataFrame(state_action_values,index = self.__GAME_STATES,columns=self.__GAME_ACTIONS)

    #gets the saved transition CSV file, named q_learning.csv or creates a new one if it was not found.
    def __get_reward_dataframe_from_csv(self):
        try:
            return pd.read_csv(self.__REWARD_CSV_PATH,index_col=0)
        except:
            return self.getNewRewardMatrix()


    def __get_policy_dataframe_from_csv(self):
        try:
            return pd.read_csv(self.__POLICY_CSV_PATH,index_col=0)
        except:
            return self.getNewPolicyMatrix()

    def loadRewardAndPolicy(self):
    
        self.__REWARD_DATAFRAME = self.__get_reward_dataframe_from_csv()
        self.__POLICY_DATAFRAME = self.__get_policy_dataframe_from_csv()


    def saveReward(self):    
        self.__REWARD_DATAFRAME.to_csv(self.__REWARD_CSV_PATH)

    def savePolicy(self):
        self.__POLICY_DATAFRAME.to_csv(self.__POLICY_CSV_PATH)


    ### GETTERS AND SETTERS
    def getENEMY_PADDLE_STATES(self):
        return self.__ENEMY_PADDLE_STATES

    def getPLAYER_PADDLE_STATES(self):
        return self.__PLAYER_PADDLE_STATES

    def setLAST_STATE(self,state):
        self.__LAST_STATE = state
    def getLAST_STATE(self):
        return self.__LAST_STATE
    def getGAME_ACTIONS(self):
        return self.__GAME_ACTIONS
    def getPOLICY_DATAFRAME(self):
        return self.__POLICY_DATAFRAME
    def getREWARD_DATAFRAME(self):
        return self.__REWARD_DATAFRAME




#DEBUGGING
#test_class = LocationModel()
#test_class.getRewardMatrix()
