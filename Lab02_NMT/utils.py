from collections import namedtuple


Task = namedtuple('Task', ['train', 'val'])




def add_to_catalog(name, catalog):
    def add_wrapper(class_to_add):
        catalog[name] = class_to_add
        return class_to_add
    return add_wrapper
