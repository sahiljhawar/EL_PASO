import importlib.util
import sys


# config.py
class Config:
    _instance = None
    _functions = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_function(cls, name, func):
        cls._functions[name] = func

    @classmethod
    def get_function(cls, name):
        return cls._functions.get(name)

    @classmethod
    def load_module_functions(cls, module_name, module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        module_attrs = {attr: getattr(module, attr) for attr in dir(module) if not attr.startswith('_')}
        for name, func in module_attrs.items():
            cls.set_function(name, func)

        print(f"Module {module_name} loaded. Contents:")
        for attr in module_attrs:
            print(f"Imported {attr} from {module_name}")


# Initialize the Config singleton
config = Config()
