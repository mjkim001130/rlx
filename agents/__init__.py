from agents.iql import IQLAgent
from agents.bc import BCAgent
from agents.sac import SACAgent
from agents.cql import CQLAgent

agents = dict(
    bc=BCAgent,
    iql=IQLAgent,
    cql=CQLAgent,
    sac=SACAgent,
)