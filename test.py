import carla_gym

#env = carla_gym.env(**kwargs)           # AEC API
if __name__ == "__main__":
    env = carla_gym.parallel_env()  # Parallel API