import sys
import time

import ale_py
import gymnasium as gym
import numpy as np
import pygame


class GamePlayer:
    def __init__(
        self,
        env: gym.Env,
        height: int,
        width: int,
        scale: int,
        tick: int,
        action_mapping,
        prevent_multi_keypress: bool = False,
        start_paused: bool = True,
    ):
        self.env = env
        self.height = height
        self.width = width
        self.scale = scale
        self.tick = tick
        self.action_mapping = action_mapping
        self.prevent_multi_keypress = prevent_multi_keypress
        self.start_paused = start_paused

        # reset env just in case it hasn't been done already
        self.env.reset()
        self.initPygame()

    def initPygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            size=(self.width * self.scale, self.height * self.scale)
        )
        self.clock = pygame.time.Clock()

    def renderGame(self):
        rgb_array = self.env.render()
        if rgb_array is not None:
            rgb_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(
                rgb_surface, (self.width * self.scale, self.height * self.scale)
            )
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.tick)

    def pauseGame(self):
        paused = True
        while paused:
            # if key pressed, break out of pause loop
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    paused = False
                    continue

    def play(self):
        running = True
        total_reward = 0.0
        reward = 0

        # game paused till first key press
        self.renderGame()
        if self.start_paused:
            self.pauseGame()

        # game loop
        while running:
            for event in pygame.event.get():
                if event.type in [pygame.QUIT, pygame.K_ESCAPE]:
                    self.env.close()
                    return

            # get key states and find corresponding actions
            keys = pygame.key.get_pressed()
            action = self.action_mapping(keys)
            if action is None:
                continue

            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            self.renderGame()
            # sleep for a short time to not register a key multiple times
            if self.prevent_multi_keypress:
                time.sleep(0.2)

            if terminated or truncated:
                running = False

        self.env.close()
        pygame.quit()
        print(f"Total reward: {total_reward}")
        print(f"Final reward: {reward}")


def play_cartpole():
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    def keyToAction(keys):
        if keys[pygame.K_LEFT]:
            action = np.int64(0)
        elif keys[pygame.K_RIGHT]:
            action = np.int64(1)
        else:
            action = env.action_space.sample()
        return action

    game = GamePlayer(
        env, width=600, height=400, scale=2, tick=15, action_mapping=keyToAction
    )
    game.play()


def play_lunar():
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    def keyToAction(keys):
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
        return [engine, lateral]

    game = GamePlayer(
        env, width=600, height=400, scale=2, tick=30, action_mapping=keyToAction
    )
    game.play()


def play_racing():
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

    def keyToAction(keys):
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
        return action

    game = GamePlayer(
        env, width=600, height=400, scale=2, tick=30, action_mapping=keyToAction
    )
    game.play()


def play_taxi():
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    def keyToAction(keys):
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
            action = None
        return action

    game = GamePlayer(
        env,
        width=550,
        height=350,
        scale=2,
        tick=30,
        prevent_multi_keypress=True,
        action_mapping=keyToAction,
    )
    game.play()


def play_pacman():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pacman-v5", obs_type="rgb", render_mode="rgb_array")

    def keyToAction(keys):
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
        return action

    game = GamePlayer(
        env, width=210, height=160, scale=4, tick=30, action_mapping=keyToAction
    )
    game.play()


def play(game):
    """Entry point for playing different games."""
    match game:
        case "cartpole":
            play_cartpole()
        case "lunar":
            play_lunar()
        case "racing":
            play_racing()
        case "taxi":
            play_taxi()
        case "pacman":
            play_pacman()
        case _:
            print("Game not recognised, exiting.")
            sys.exit(1)
