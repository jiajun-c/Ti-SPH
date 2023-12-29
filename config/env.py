# the configuration of the environment
import numpy as np

class env_config:
    def __init__(self, f) -> None:
        self.config = f['config']
        self.dim = self.config['dim']
        self.dominStart = np.array(self.config['dominStart'])
        self.dominEnd = np.array(self.config['dominEnd'])
        self.dt = self.config['dt']
        self.dominSize = self.dominEnd - self.dominStart

