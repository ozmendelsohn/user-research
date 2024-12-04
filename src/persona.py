from dataclasses import dataclass
from typing import List

@dataclass
class Persona:
    """
    Represents a user persona for product research.
    
    Parameters
    ----------
    name : str
        Name of the persona
    age : int
        Age of the persona
    occupation : str
        Occupation or professional background
    interests : List[str]
        Key interests and hobbies
    pain_points : List[str]
        Specific challenges or needs
    tech_savviness : int
        Rating from 1-5 of technical proficiency
    background : str
        Detailed background story
    """
    name: str
    age: int
    occupation: str
    interests: List[str]
    pain_points: List[str]
    tech_savviness: int
    background: str 