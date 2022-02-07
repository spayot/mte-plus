import time

class Repr:
    def __repr__(self):
        return self.__class__.__name__ + "()"
    
    
def set_repr(obj, attributes: list[str]):
    att_reprs = [f"{att}={getattr(obj, att)}" for att in attributes]
    
    return obj.__class__.__name__ + f'({", ".join(att_reprs)})'

def convert_time(seconds: int):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)