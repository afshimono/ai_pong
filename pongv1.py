import pygame
import sys
import time
import random
import math
from pygame.locals import *
import q_learning as artificialIntelligenceModule
import location_direction_model as modelImplementation


# Number of frames per second
# Change this value to speed up or slow down your game
FPS = 3000

#Global Variables to be used in our program

WINDOWWIDTH = 640
WINDOWHEIGHT = 480
LINETHICKNESS = 8
PADDLESIZE = 60
PADDLEOFFSET = 20
SCOREWINNER = 11
WINNER = False
BASICFONTSIZE = 20
Q_LEARNING = True
HUMAN_PLAYER = False
CONTINUES = 1000
model = modelImplementation.LocationDirectionModel(WINDOWWIDTH, WINDOWHEIGHT,LINETHICKNESS,PADDLEOFFSET,PADDLESIZE,SCOREWINNER)
q_learn = artificialIntelligenceModule.QLearn(model)


# Set up the colours
BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)

#game working params
ballDirX = 4*random.choice([-1, 1])  
ballDirY = 4*random.choice([-1, 1]) 
ballX = WINDOWWIDTH/2 - LINETHICKNESS/2
ballY = WINDOWHEIGHT/2 - LINETHICKNESS/2
playerOnePosition = (WINDOWHEIGHT - PADDLESIZE) /2
playerTwoPosition = (WINDOWHEIGHT - PADDLESIZE) /2
playerOneLastPosition = playerOnePosition
playerTwoLastPosition = playerTwoPosition
score = [0,0]
ballDistanceY = 0.0


#Draws the arena the game will be played in. 
def drawArena():
    DISPLAYSURF.fill((0,0,0))
    #Draw outline of arena
    pygame.draw.rect(DISPLAYSURF, WHITE, ((0,0),(WINDOWWIDTH,WINDOWHEIGHT)), LINETHICKNESS)
    #Draw centre line
    pygame.draw.line(DISPLAYSURF, WHITE, (int((WINDOWWIDTH/2)),0),(int(WINDOWWIDTH/2),int(WINDOWHEIGHT)), int(LINETHICKNESS/4))


#Draws the paddle
def drawPaddle(paddle):
    #Stops paddle moving too low
    if paddle.bottom > WINDOWHEIGHT - LINETHICKNESS:
        paddle.bottom = WINDOWHEIGHT - LINETHICKNESS
    #Stops paddle moving too high
    elif paddle.top < LINETHICKNESS:
        paddle.top = LINETHICKNESS
    #Draws paddle
    pygame.draw.rect(DISPLAYSURF, WHITE, paddle)


#draws the ball
def drawBall(ball):
    pygame.draw.rect(DISPLAYSURF, WHITE, ball)

#moves the ball returns new position
def moveBall(ball, ballDirX, ballDirY, ballDistanceY):
    ball.x += int(ballDirX)
    ball.y += int(math.floor(ballDirY + ballDistanceY))
    ballDistanceY += ballDirY % 1
    ballDistanceY = ballDistanceY % 1
    return [ball,ballDistanceY]


#Draw Arena winner
def drawWinner(score):
    # print("->" + str(score[1]))
    CoorX = 0
    CoorY = 0
    global WINNER
    if WINNER == True:
        if score[0] == SCOREWINNER:
            CoorX = WINDOWWIDTH/8 + WINDOWWIDTH/16
            CoorY = WINDOWHEIGHT/8 
        if score[1] == SCOREWINNER:
            CoorX = 6*WINDOWWIDTH/8 - WINDOWWIDTH/16
            CoorY = WINDOWHEIGHT/8
                  
        resultSurfP1 = BASICFONT.render('WINNER', True, WHITE)
        resultRectP1 = resultSurfP1.get_rect()
        resultRectP1.topleft = (CoorX, CoorY)
        DISPLAYSURF.blit(resultSurfP1, resultRectP1) 
        pygame.display.update()
        #time.sleep( 2 )

        
    

#Checks for a collision with a wall, and 'bounces' ball off it.
#Returns new direction
def checkEdgeCollision(ball, ballDirX, ballDirY):
    if ball.top == (LINETHICKNESS) or ball.bottom == (WINDOWHEIGHT - LINETHICKNESS):
        ballDirY = ballDirY * -1.0
    return ballDirX, ballDirY

#Checks is the ball has hit a paddle, and 'bounces' ball off it.     
def checkHitBall(ball, paddle1, paddle2, ballDirX, ballDirY,playerOneLastPosition,playerTwoLastPosition,ballDistanceY):
    if ballDirX == -4 and paddle1.right == ball.left and paddle1.top <= ball.bottom and paddle1.bottom >= ball.top:
        ballDirX = 4 
        if (paddle1.y - playerOneLastPosition) < 0:
            if ballDirY>1.3 or ballDirY <= 2:
                if ballDirY > -6:
                    ballDirY -= 2
                else: 
                    ballDirY = -6
            else:
                ballDirY = 0
        elif (paddle1.y - playerOneLastPosition) > 0:
            if ballDirY<0.3 or ballDirY>=2:
                if ballDirY < 6:
                    ballDirY += 2
                else:
                    ballDirY = 6
            else:
                ballDirY = 0           
        ballDistanceY = 0

        print('Hit! State: ' + model.getGameState(ball.x,ball.y,paddle1.y,paddle2.y,ballDirX,ballDirY) +'  Model_state:  '+ q_learn.getCurrentState() + '  Reward_matrix:  ')
        print(model.getREWARD_DATAFRAME().loc[q_learn.getCurrentState()])
        
    elif ballDirX == 4 and paddle2.left == ball.right and paddle2.top <= ball.bottom and paddle2.bottom >= ball.top:
        ballDirX = -4
        if (paddle2.y - playerTwoLastPosition) < 0:
            if ballDirY>1.3 or ballDirY <= 2:
                if ballDirY > -6:
                    ballDirY -= 2
                else:
                    ballDirY = -6
            else:
                ballDirY = 0
        elif (paddle2.y - playerTwoLastPosition) > 0:
            if ballDirY<0.3 or ballDirY>=2:
                if ballDirY < 6:
                    ballDirY += 2
                else:
                    ballDirY = 6
            else:
                ballDirY = 0    
        ballDistanceY = 0
    
    return ballDirX, ballDirY, ballDistanceY

#Checks to see if a point has been scored returns new score
def checkPointScored(paddle1, ball, score, ballDirX,ballDirY):
    #reset points if left wall is hit
    if ball.left == LINETHICKNESS: 
        score[1] = score[1] + 1
        ballX = WINDOWWIDTH/2 - LINETHICKNESS/2
        ballY = WINDOWHEIGHT/2 - LINETHICKNESS/2
        ballDirX = -4
        ballDirY = 4*random.choice([-1, 1]) 
        ball = pygame.Rect(ballX, ballY, LINETHICKNESS, LINETHICKNESS)
        return (score,ball,ballDirX,ballDirY)
    
    elif ball.right == WINDOWWIDTH - LINETHICKNESS:
        score[0] = score[0] + 1
        ballX = WINDOWWIDTH/2 - LINETHICKNESS/2
        ballY = WINDOWHEIGHT/2 - LINETHICKNESS/2
        ballDirX = 4
        ballDirY = 4*random.choice([-1, 1])
        ball = pygame.Rect(ballX, ballY, LINETHICKNESS, LINETHICKNESS)
        return (score,ball,ballDirX,ballDirY)
    #if no points scored, return score unchanged
    else: return (score,ball,ballDirX,ballDirY)

#Checks to see if the game has ended
def checkGameOver(score):
    global WINNER

    if score[0]== SCOREWINNER or score[1]== SCOREWINNER:
        WINNER = True        
        drawWinner(score)
                  
    return score

#Artificial Intelligence of computer player 
def artificialIntelligence(ball, ballDirX, paddle2):
    #If ball is moving away from paddle, center bat
    if ballDirX == -4:
        if paddle2.centery < (WINDOWHEIGHT/2):
            paddle2.y += 4
        elif paddle2.centery > (WINDOWHEIGHT/2):
            paddle2.y -= 4
    #if ball moving towards bat, track its movement. 
    elif ballDirX == 4:
        if paddle2.centery < ball.centery:
            paddle2.y += 4
        else:
            paddle2.y -=4
        
    return paddle2

#Displays the current score on the screen
def displayScore(score,ballDirY):
    resultSurfP1 = BASICFONT.render('Player 1 = %s' %(score[0]), True, WHITE)
    resultRectP1 = resultSurfP1.get_rect()
    resultRectP1.topleft = (WINDOWWIDTH/8, 25)
    DISPLAYSURF.blit(resultSurfP1, resultRectP1)
    #resultSurfP2 = BASICFONT.render('Computer = %s' %(score[1]), True, WHITE)
    resultSurfP2 = BASICFONT.render('Player 2 = %s' %(score[1]), True, WHITE)
    resultRectP2 = resultSurfP2.get_rect()
    resultRectP2.topright = (7*WINDOWWIDTH/8, 25)
    DISPLAYSURF.blit(resultSurfP2, resultRectP2)





#Main function
def play():
    pygame.init()
    #load global variables
    global ballDirX, ballDirY, ballX, ballY, playerOnePosition, playerTwoPosition, playerOneLastPosition, WINNER
    global playerTwoLastPosition, score, ballDistanceY, BASICFONT, Q_LEARNING, DISPLAYSURF, paddle1, paddle2, ball
    ##Font information
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

    global PLAYING
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH,WINDOWHEIGHT)) 
    pygame.display.set_caption('Pong')


   
    #Creates Rectangles for ball and paddles.
    paddle1 = pygame.Rect(PADDLEOFFSET,random.choice([0,WINDOWHEIGHT]), LINETHICKNESS,PADDLESIZE)
    paddle2 = pygame.Rect(WINDOWWIDTH - PADDLEOFFSET - LINETHICKNESS, playerTwoPosition, LINETHICKNESS,PADDLESIZE)
    ball = pygame.Rect(ballX, ballY, LINETHICKNESS, LINETHICKNESS)

    #Draws the starting position of the Arena
    drawArena()
    drawPaddle(paddle1)
    drawPaddle(paddle2)
    drawBall(ball)

    pygame.mouse.set_visible(0) # make cursor invisible
    
    
    while True: #main game loop
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        # mouse movement commands
        if HUMAN_PLAYER:
            keys=pygame.key.get_pressed()
            if keys[pygame.K_DOWN] and paddle1.y < (WINDOWHEIGHT - PADDLESIZE):
                paddle1.y += 4
            elif keys[pygame.K_UP] and paddle1.y > 0:
                paddle1.y -= 4
        else:
            move = q_learn.next_move(ball.x,ball.y,paddle1.y,paddle2.y,ballDirX,ballDirY)
            if move[0] == 'UP' and paddle1.y > 0:
                #print('Moving Up!')
                paddle1.y -= 4
            elif move[0] == 'DOWN' and paddle1.y > 0:
                paddle1.y += 4
                

        drawArena()
        drawPaddle(paddle1)
        drawPaddle(paddle2)
        drawBall(ball)


        ball,ballDistanceY = moveBall(ball, ballDirX, ballDirY, ballDistanceY)
        ballDirX, ballDirY = checkEdgeCollision(ball, ballDirX, ballDirY)
        score, ball,ballDirX,ballDirY = checkPointScored(paddle1, ball, score, ballDirX,ballDirY)
        ballDirX,ballDirY,ballDistanceY = checkHitBall(ball, paddle1, paddle2, ballDirX, ballDirY,playerOneLastPosition,playerTwoLastPosition,ballDistanceY)
        paddle2 = artificialIntelligence(ball, ballDirX, paddle2)
        score = checkGameOver(score)
        
        if WINNER:
            WINNER = False
            return None
        displayScore(score,ballDirY)
        

        pygame.display.update()
        #FPSCLOCK.tick(FPS)

        #updates last position of paddles
        playerOneLastPosition = paddle1.y
        playerTwoLastPosition = paddle2.y

        #runs learning
        if Q_LEARNING:
            q_learn.learn()

    #pygame.quit()
    
def main():
    global CONTINUES, score
    
    while CONTINUES >= 0:
        print('Continues:   '+str(CONTINUES))
        play()
        model.savePolicy()
        i_won = 0
        if score[0] == 11:
            i_won = 1
        else:
            i_won = 0       
        q_learn.incrementEpoch(score[0],i_won)
        score = [0,0]
        CONTINUES = CONTINUES - 1
    q_learn.plot_data()

def debug():
    #q_learn.incrementEpoch(2,1)
    q_learn.plot_data()
    #q_learn.next_move(ballX,ballY,playerOnePosition,playerTwoPosition)

if __name__=='__main__':
    #debug()
    main()