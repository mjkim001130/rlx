
from agents.iql import IQLAgent
from agents.bc import BCAgent
from agents.sac import SACAgent

agents = dict(
    bc=BCAgent,
    iql=IQLAgent,
    sac=SACAgent,
)