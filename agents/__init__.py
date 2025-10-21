from agents.iql import IQLAgent
from agents.bc import BCAgent
from agents.cql import CQLAgent
from agents.sac import SACAgent

agents = dict(
    bc=BCAgent,
    iql=IQLAgent,
    cql=CQLAgent,
    sac=SACAgent,
)