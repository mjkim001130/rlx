from agents.iql import IQLAgent
from agents.bc import BCAgent
from agents.sac import SACAgent
from agents.cql import CQLAgent
from agents.td3_bc import TD3BCAgent

agents = dict(
    bc=BCAgent,
    iql=IQLAgent,
    cql=CQLAgent,
    sac=SACAgent,
    td3_bc=TD3BCAgent,
)
