# 贪吃蛇

贪吃蛇是一个起源于1976年的街机游戏 Blockade，玩家控制蛇上下左右吃到食物并将身体增长，吃到食物后移动速度逐渐加快，直到碰到墙体或者蛇的身体算游戏结束。

![image-20200901202636603](img/image-20200901202636603.png)

如图，本次任务整个游戏版面大小为560X560，绿色部分就是我们的智能体贪吃蛇，红色方块就是食物，墙位于四周，一旦食物被吃掉，会在下一个随机位置刷出新的食物。蛇的每一节以及食物的大小为40X40，除开墙体(厚度也为40)，蛇可以活动的范围为480X480，也就是12X12的栅格。环境的状态等信息如下：

* state：为一个元组，包含(adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right). 

  * [adjoining_wall_x, adjoining_wall_y]：提供蛇头是否与墙体相邻的信息，具体包含9个状态

    adjoining_wall_x：0表示x轴方向蛇头无墙体相邻，1表示有墙在蛇头左边，2表示有墙在右边adjoining_wall_y：0表示y轴方向蛇头无墙体相邻，1表示有墙在蛇头上边，2表示有墙在下边

    注意[0,0]也包括蛇跑出480X480范围的情况

  * [food_dir_x, food_dir_y]：表示食物与蛇头的位置关系

    food_dir_x：0表示食物与蛇头同在x轴上，1表示食物在蛇头左侧(不一定相邻)，2表示在右边

    food_dir_y：0表示食物与蛇头同在y轴上，1表示食物在蛇头上面，2表示在下面

  * [adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]：用以检查蛇的身体是否在蛇头的附近

    adjoining_body_top：1表示蛇头上边有蛇的身体，0表示没有

    adjoining_body_bottom：1表示蛇头下边有蛇的身体，0表示没有

    adjoining_body_left：1表示蛇头左边有蛇的身体，0表示没有

    adjoining_body_right：1表示蛇头右边有蛇的身体，0表示没有

* action：即上下左右

* reward：如果吃到食物给一个+1的reward，如果蛇没了就-1，其他情况给-0.1的reward



