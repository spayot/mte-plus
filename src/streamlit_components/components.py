import pandas as pd
import streamlit as st

class StreamlitLogger:
    def __init__(self, height: int = 10):
        """logging box"""
        self.logtxtbox = st.empty()
        self.height = height
        self.text = []
        
    def log(self, logtext: str) -> None:
        self.text.append(logtext)
        # self.logtxtbox.text_area("Logging: ",'\n'.join(self.text), height=self.height)
        self.logtxtbox.text("    Logging :" + logtext)
        
        
def create_checkbox_list(loc, options: dict, title: str) -> list:
    loc.write('## ' + title)
    items = [item 
             for name, item in options.items() 
             if loc.checkbox(name, value=True)]            
    return items