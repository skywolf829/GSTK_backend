from abc import abstractmethod

import pkgutil
import importlib
import inspect

def import_and_read_keys():
    keys = {}
    package = importlib.import_module("edits")
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_module_name = f"{'edits'}.{name}"
        module = importlib.import_module(full_module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Filter out classes not defined in this module (to avoid imported ones)
            if obj.__module__ == module.__name__:
                print(full_module_name)
                try:
                    key_value = obj.key
                    keys[name] = key_value
                except AttributeError:
                    print(f"Class {name} in module {module.__name__} does not have a 'key' attribute.")
    return keys

class EditCommand():
    
    def __init__(self, model, renderer, dataset, trainer, settings):
        self.model = model
        self.dataset = dataset
        self.renderer = renderer
        self.trainer = trainer
        self.settings = settings
    
        self.completed = False
        self.key = ""

    @abstractmethod
    def execute(self):
        pass
    