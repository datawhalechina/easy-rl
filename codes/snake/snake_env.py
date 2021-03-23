import random
import pygame
import utils

class SnakeEnv:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y):
        self.game = Snake(snake_head_x, snake_head_y, food_x, food_y)
        self.render = False

    def get_actions(self):
        return self.game.get_actions()

    def reset(self):
        return self.game.reset()
    
    def get_points(self):
        return self.game.get_points()

    def get_state(self):
        return self.game.get_state()

    def step(self, action):
        state, points, dead = self.game.step(action)
        if self.render:
            self.draw(state, points, dead)
        # return state, reward, done
        return state, points, dead

    def draw(self, state, points, dead):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        self.display.fill(utils.BLUE)    
        pygame.draw.rect( self.display, utils.BLACK,
                [
                    utils.GRID_SIZE,
                    utils.GRID_SIZE,
                    utils.DISPLAY_SIZE - utils.GRID_SIZE * 2,
                    utils.DISPLAY_SIZE - utils.GRID_SIZE * 2
                ])

        # draw snake head
        pygame.draw.rect(
                    self.display, 
                    utils.GREEN,
                    [
                        snake_head_x,
                        snake_head_y,
                        utils.GRID_SIZE,
                        utils.GRID_SIZE
                    ],
                    3
                )
        # draw snake body
        for seg in snake_body:
            pygame.draw.rect(
                self.display, 
                utils.GREEN,
                [
                    seg[0],
                    seg[1],
                    utils.GRID_SIZE,
                    utils.GRID_SIZE,
                ],
                1
            )
        # draw food
        pygame.draw.rect(
                    self.display, 
                    utils.RED,
                    [
                        food_x,
                        food_y,
                        utils.GRID_SIZE,
                        utils.GRID_SIZE
                    ]
                )

        text_surface = self.font.render("Points: " + str(points), True, utils.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = ((280),(25))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()
        if dead:
            # slow clock if dead
            self.clock.tick(1)
        else:
            self.clock.tick(5)

        return 


    def display(self):
        pygame.init()
        pygame.display.set_caption('MP4: Snake')
        self.clock = pygame.time.Clock()
        pygame.font.init()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        self.display = pygame.display.set_mode((utils.DISPLAY_SIZE, utils.DISPLAY_SIZE), pygame.HWSURFACE)
        self.draw(self.game.get_state(), self.game.get_points(), False)
        self.render = True
            
class Snake:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y):
        self.init_snake_head_x,self.init_snake_head_y = snake_head_x,snake_head_y # 蛇头初始位置
        self.init_food_x, self.init_food_y = food_x, food_y # 食物初始位置
        self.reset()

    def reset(self):
        self.points = 0
        self.snake_head_x, self.snake_head_y = self.init_snake_head_x, self.init_snake_head_y
        self.food_x, self.food_y = self.init_food_x, self.init_food_y
        self.snake_body = [] # 蛇身的位置集合

    def get_points(self):
        return self.points

    def get_actions(self):
        return [0, 1, 2, 3]

    def get_state(self):
        return [
            self.snake_head_x,
            self.snake_head_y,
            self.snake_body,
            self.food_x,
            self.food_y
        ]

    def move(self, action):
        '''根据action指令移动蛇头，并返回是否撞死
        '''
        delta_x = delta_y = 0
        if action == 0: # 上
            delta_x = utils.GRID_SIZE
        elif action == 1:
            delta_x = - utils.GRID_SIZE
        elif action == 2:
            delta_y = - utils.GRID_SIZE
        elif action == 3:
            delta_y = utils.GRID_SIZE
        old_body_head = None
        if len(self.snake_body) == 1:
            old_body_head = self.snake_body[0]

        self.snake_body.append((self.snake_head_x, self.snake_head_y))
        self.snake_head_x += delta_x
        self.snake_head_y += delta_y

        if len(self.snake_body) > self.points: # 说明没有吃到食物
            del(self.snake_body[0])

        self.handle_eatfood()

        # 蛇长大于1时，蛇头与蛇身任一位置重叠则看作蛇与自身相撞
        if len(self.snake_body) >= 1:
            for seg in self.snake_body:
                if self.snake_head_x == seg[0] and self.snake_head_y == seg[1]:
                    return True

        # 蛇长为1时，如果蛇头与之前的位置重复则看作蛇与自身相撞
        if len(self.snake_body) == 1:
            if old_body_head == (self.snake_head_x, self.snake_head_y):
                return True

        # 蛇头是否撞墙
        if (self.snake_head_x < utils.GRID_SIZE or self.snake_head_y < utils.GRID_SIZE or
            self.snake_head_x + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE or self.snake_head_y + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE):
            return True

        return False

    def step(self, action):
        is_dead = self.move(action)
        return self.get_state(), self.get_points(), is_dead

    def handle_eatfood(self):
        if (self.snake_head_x == self.food_x) and (self.snake_head_y == self.food_y):
            self.random_food()
            self.points += 1

    def random_food(self):
        '''生成随机位置的食物
        '''
        max_x = (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE)
        max_y = (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE)
        
        self.food_x = random.randint(utils.WALL_SIZE, max_x)//utils.GRID_SIZE * utils.GRID_SIZE
        self.food_y = random.randint(utils.WALL_SIZE, max_y)//utils.GRID_SIZE * utils.GRID_SIZE

        while self.check_food_on_snake(): # 食物不能生成在蛇身上
            self.food_x = random.randint(utils.WALL_SIZE, max_x)//utils.GRID_SIZE * utils.GRID_SIZE
            self.food_y = random.randint(utils.WALL_SIZE, max_y)//utils.GRID_SIZE * utils.GRID_SIZE

    def check_food_on_snake(self):
        if self.food_x == self.snake_head_x and self.food_y == self.snake_head_y:
            return True 
        for seg in self.snake_body:
            if self.food_x == seg[0] and self.food_y == seg[1]:
                return True
        return False
        
    
