# simulations/disease_sim/systems.py
"""
Defines the logic systems for the Disease simulation.
"""

import random
from typing import Any, Dict, List, Type, cast
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System
from .components import (
    DiseaseStateComponent,
    DiseaseStateEnum,
    DiseaseParametersComponent,
    SocialNetworkComponent,
)
from .providers import SocialContactProviderInterface


class DiseaseTransmissionSystem(System):
    """
    The primary engine of infection spread, based on the SEIR model.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        DiseaseStateComponent,
        DiseaseParametersComponent,
        SocialNetworkComponent,
    ]

    def __init__(
        self,
        simulation_state: Any,
        config: Any,
        cognitive_scaffold: Any,
        contact_provider: SocialContactProviderInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.contact_provider = contact_provider
        self.infection_prob_i = self.config.disease.infection_probability_i
        self.infection_prob_e = self.config.disease.infection_probability_e

    async def update(self, current_tick: int) -> None:
        infectious_agents = []
        exposed_agents = []
        all_agents_with_state = self.simulation_state.get_entities_with_components(
            [DiseaseStateComponent]
        )

        for agent_id, components in all_agents_with_state.items():
            state_comp = cast(
                DiseaseStateComponent, components.get(DiseaseStateComponent)
            )
            if state_comp.state == DiseaseStateEnum.INFECTIOUS:
                infectious_agents.append(agent_id)
            elif state_comp.state == DiseaseStateEnum.EXPOSED:
                exposed_agents.append(agent_id)

        # Process transmissions from Infectious agents
        for agent_id in infectious_agents:
            await self._process_transmission(
                agent_id, self.infection_prob_i, all_agents_with_state
            )

        # Process transmissions from Exposed agents
        for agent_id in exposed_agents:
            await self._process_transmission(
                agent_id, self.infection_prob_e, all_agents_with_state
            )

    async def _process_transmission(
        self, agent_id: str, prob: float, all_agents: Dict[str, Any]
    ):
        contacts = self.contact_provider.get_contacts(agent_id)
        for contact_id in contacts:
            contact_components = all_agents.get(contact_id)
            if not contact_components:
                continue

            contact_state_comp = cast(
                DiseaseStateComponent, contact_components.get(DiseaseStateComponent)
            )
            params_comp = cast(
                DiseaseParametersComponent,
                self.simulation_state.get_component(
                    contact_id, DiseaseParametersComponent
                ),
            )

            if contact_state_comp.state == DiseaseStateEnum.SUSCEPTIBLE:
                if random.random() < prob:
                    contact_state_comp.state = DiseaseStateEnum.EXPOSED
                    contact_state_comp.incubation_timer = params_comp.incubation_period


class DiseaseProgressionSystem(System):
    """Manages the time-based progression of the disease within agents."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        DiseaseStateComponent,
        DiseaseParametersComponent,
    ]

    async def update(self, current_tick: int) -> None:
        agents_to_progress = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )

        for _, components in agents_to_progress.items():
            state_comp = cast(
                DiseaseStateComponent, components.get(DiseaseStateComponent)
            )
            params_comp = cast(
                DiseaseParametersComponent, components.get(DiseaseParametersComponent)
            )

            if state_comp.state == DiseaseStateEnum.EXPOSED:
                state_comp.incubation_timer -= 1
                if state_comp.incubation_timer <= 0:
                    state_comp.state = DiseaseStateEnum.INFECTIOUS
                    state_comp.infection_timer = params_comp.infection_period
            elif state_comp.state == DiseaseStateEnum.INFECTIOUS:
                state_comp.infection_timer -= 1
                if state_comp.infection_timer <= 0:
                    state_comp.state = DiseaseStateEnum.REMOVED


class InterventionSystem(System):
    """Models the effects of external policy interventions."""

    REQUIRED_COMPONENTS: List[Type[Component]] = []

    def __init__(self, simulation_state: Any, config: Any, cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.lockdown_tick = self.config.interventions.lockdown_tick
        self.lockdown_triggered = False

    async def update(self, current_tick: int) -> None:
        if self.lockdown_tick is None or self.lockdown_triggered:
            return

        if current_tick >= self.lockdown_tick:
            print(
                f"--- INTERVENTION: Lockdown policy enacted at tick {current_tick} ---"
            )
            payload = {
                "effectiveness": self.config.interventions.lockdown_effectiveness
            }
            if self.event_bus:
                self.event_bus.publish("lockdown_policy_enacted", payload)
            self.lockdown_triggered = True
