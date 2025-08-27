"""
NOTE: I KNOW THIS SCRIPT HAS A BUNCH OF DUPLICATED CODE BUT THIS JUST MAKES THE
GAMES RUN BETTER IF I CAN HANDLE ALL THE DATA CREATION/MODIFICATION IN A SINGLE
BLOCK. AND ALSO EASIER TO HANDLE MINUTE DIFFERENCES IN THE GAMES.
"""

import sys
import time

import ale_py
import gymnasium as gym
import numpy as np
import pygame


def init_pygame(width, height):
    pygame.init()
    screen = pygame.display.set_mode(size=(width, height))
    clock = pygame.time.Clock()
    return screen, clock


def render_game(env, screen, clock, height, width, scale, tick):
    rgb_array = env.render()
    if rgb_array is not None:
        rgb_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(
            rgb_surface, (width * scale, height * scale)
        )
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(tick)


def pause_game():
    paused = True
    while paused:
        # if key pressed, break out of pause loop
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                paused = False
                continue


def cart_pole(width=600, height=400, scale=2):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    observation, info = env.reset()

    screen, clock = init_pygame(width * scale, height * scale)
    running = True
    total_reward = 0.0
    reward = 0

    # game paused till first key press
    render_game(env, screen, clock, height, width, scale, tick=1)
    pause_game()

    # game loop
    while running:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                env.close()
                return

        # get key states
        keys = pygame.key.get_pressed()

        # convert key to action
        if keys[pygame.K_LEFT]:
            action = np.int64(0)
        elif keys[pygame.K_RIGHT]:
            action = np.int64(1)
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        render_game(env, screen, clock, height, width, scale, tick=15)

        if terminated or truncated:
            running = False

    env.close()
    pygame.quit()
    print(f"Total reward: {total_reward}")
    print(f"Final reward: {reward}")


def lunar_lander(width=600, height=400, scale=2):
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    observation, info = env.reset()
    action = np.array([0.0, 0.0], dtype=np.float32)

    screen, clock = init_pygame(width * scale, height * scale)
    running = True
    total_reward = 0.0
    reward = 0

    # game paused till first key press
    render_game(env, screen, clock, height, width, scale, tick=1)
    pause_game()

    # game loop
    while running:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                env.close()
                return

        # get key states
        keys = pygame.key.get_pressed()

        # convert key to action
        if keys[pygame.K_LEFT]:
            lateral = -1
        elif keys[pygame.K_RIGHT]:
            lateral = 1
        else:
            lateral = 0

        if keys[pygame.K_UP]:
            engine = 1
        else:
            engine = 0

        action = [engine, lateral]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        render_game(env, screen, clock, height, width, scale, tick=30)

        if terminated or truncated:
            running = False

    env.close()
    pygame.quit()
    print(f"Total reward: {total_reward}")
    print(f"Final reward: {reward}")


def car_racing(width=600, height=400, scale=2):
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    observation, info = env.reset()

    screen, clock = init_pygame(width * scale, height * scale)
    running = True
    total_reward = 0.0
    reward = 0

    # game paused till first key press
    render_game(env, screen, clock, height, width, scale, tick=1)
    pause_game()

    # game loop
    while running:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                env.close()
                return

        # get key states
        keys = pygame.key.get_pressed()

        # convert key to action
        if keys[pygame.K_RIGHT]:
            action = np.int64(1)
        elif keys[pygame.K_LEFT]:
            action = np.int64(2)
        elif keys[pygame.K_UP]:
            action = np.int64(3)
        elif keys[pygame.K_DOWN]:
            action = np.int64(4)
        else:
            action = np.int64(0)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        render_game(env, screen, clock, height, width, scale, tick=30)

        if terminated or truncated:
            running = False

    env.close()
    pygame.quit()
    print(f"Total reward: {total_reward}")
    print(f"Final reward: {reward}")


def taxi(width=550, height=350, scale=2):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    observation, info = env.reset()

    screen, clock = init_pygame(width * scale, height * scale)
    running = True
    total_reward = 0.0
    reward = 0

    # render initial state. game does nothing if not key presses so it'll stay
    # stuck on this frame
    render_game(env, screen, clock, height, width, scale, tick=30)

    # game loop
    while running:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                env.close()
                return

        # get key states
        keys = pygame.key.get_pressed()

        # convert key to action
        if keys[pygame.K_DOWN]:
            action = np.int64(0)
        elif keys[pygame.K_UP]:
            action = np.int64(1)
        elif keys[pygame.K_RIGHT]:
            action = np.int64(2)
        elif keys[pygame.K_LEFT]:
            action = np.int64(3)
        elif keys[pygame.K_1]:
            action = np.int64(4)
        elif keys[pygame.K_2]:
            action = np.int64(5)
        else:
            continue

        observation, reward, terminated, truncated, info = env.step(action)
        # sleep timer to not register multiple keypresses randomly
        # extra key presses matter a lot for this game
        time.sleep(0.2)
        total_reward += reward

        render_game(env, screen, clock, height, width, scale, tick=30)

        if terminated or truncated:
            running = False

    env.close()
    pygame.quit()
    print(f"Total reward: {total_reward}")
    print(f"Final reward: {reward}")


def pacman(width=250, height=160, scale=4):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pacman-v5", obs_type="rgb", render_mode="rgb_array")
    observation, info = env.reset()

    screen, clock = init_pygame(width * scale, height * scale)
    running = True
    total_reward = 0.0
    reward = 0

    # game loop
    while running:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                env.close()
                return

        # get key states
        keys = pygame.key.get_pressed()

        # convert key to action
        if keys[pygame.K_UP]:
            action = np.int64(1)
        elif keys[pygame.K_RIGHT]:
            action = np.int64(2)
        elif keys[pygame.K_LEFT]:
            action = np.int64(3)
        elif keys[pygame.K_DOWN]:
            action = np.int64(4)
        else:
            action = np.int64(0)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        render_game(env, screen, clock, height, width, scale, tick=30)

        if terminated or truncated:
            running = False

    env.close()
    pygame.quit()
    print(f"Total reward: {total_reward}")
    print(f"Final reward: {reward}")


def play(game):
    """Entry point for playing different games."""
    match game:
        case "cartpole":
            cart_pole()
        case "lunar":
            lunar_lander()
        case "racing":
            car_racing()
        case "taxi":
            taxi()
        case "pacman":
            pacman()
        case _:
            print("Game not recognised, exiting.")
            sys.exit(1)
