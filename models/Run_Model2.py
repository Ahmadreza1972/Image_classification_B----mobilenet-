import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model2 import Config
from models.Base_model import BaseModle

class ModelProcess(BaseModle):
    def __init__(self):
        self._config=Config()
        super().__init__(self._config,"model2")
        
model=ModelProcess()
model.main()

