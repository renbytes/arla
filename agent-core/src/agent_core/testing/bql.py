# src/agent_core/testing/bql.py

from typing import Any, Callable, Dict, List, Optional, Type
from agent_core.core.ecs.component import Component

# --- Type Definitions for the DSL ---
ConditionFunction = Callable[[str, str, Any], str]
TemporalCondition = str # Will hold 'ALWAYS', 'EVENTUALLY', 'AT_TICK', etc.

class AttributeCondition:
    """Represents a single atomic condition on a Component attribute."""
    def __init__(self, component_type: Type[Component], attribute_name: str, operator: str, value: Any) -> None:
        self.component_type = component_type
        self.attribute_name = attribute_name
        self.operator = operator
        self.value = value

    def __str__(self) -> str:
        """Translates the condition into a SQL-like fragment."""
        comp_name = self.component_type.__name__
        # Assumes component data is stored in a JSONB column named 'components_data'
        # The path is: components_data -> 'ComponentClassName' -> 'attributeName'
        path = f"components_data->'{comp_name}'->>'{self.attribute_name}'"
        
        # Need to handle quoting based on value type for correct SQL generation
        if isinstance(self.value, str):
            value_str = f"'{self.value}'"
            # NOTE: Numeric fields must be explicitly cast in SQL from JSONB for comparison
            # Example: (components_data->'EnergyComponent'->>'current_energy')::FLOAT > 0.0
            return f"({path})::TEXT {self.operator} {value_str}"
        elif isinstance(self.value, (int, float)):
            value_str = str(self.value)
            return f"({path})::FLOAT {self.operator} {value_str}"
        else:
            return f"({path}) {self.operator} '{self.value}'"


class BehavioralAssertion:
    """Represents a full behavioral hypothesis with a temporal scope."""
    def __init__(self, temporal: TemporalCondition, conditions: List[AttributeCondition]) -> None:
        self.temporal = temporal
        self.conditions = conditions

    def to_sql_where_clause(self) -> str:
        """Combines all conditions into a single SQL WHERE clause."""
        return " AND ".join(str(c) for c in self.conditions)

    def __str__(self) -> str:
        """A human-readable representation of the assertion."""
        conditions_str = " AND ".join(
            f"{c.component_type.__name__}.{c.attribute_name} {c.operator} {c.value}"
            for c in self.conditions
        )
        return f"ASSERT {self.temporal}: {conditions_str}"

# --- DSL Functions (The User-Facing API) ---

def _build_condition(component_type: Type[Component]) -> Callable[[str, str, Any], AttributeCondition]:
    """Factory to create condition builders for a specific Component type."""

    def builder(attribute_name: str, operator: str, value: Any) -> AttributeCondition:
        """Validates attribute and creates the condition object."""
        if attribute_name not in component_type._get_component_fields(component_type):
            raise ValueError(f"Attribute '{attribute_name}' not found in {component_type.__name__}. Available fields: {list(component_type._get_component_fields(component_type).keys())}")
        
        return AttributeCondition(component_type, attribute_name, operator, value)

    return builder

# The central BQL object exposed to users
class BQL:
    """
    The main Behavioral Query Language (BQL) object.
    It provides methods to define temporal assertions and access Component attributes.
    """
    
    def __init__(self, component_map: Dict[str, Type[Component]]):
        self.component_map = component_map
        self.assertions: List[BehavioralAssertion] = []
        
        # Dynamically create component accessors (e.g., BQL.EnergyComponent)
        for name, comp_type in component_map.items():
            setattr(self, name, _ComponentAccessor(comp_type))
            
    def assert_always(self, *conditions: AttributeCondition) -> BehavioralAssertion:
        """Asserts that the conditions must be true for every single tick."""
        assertion = BehavioralAssertion("ALWAYS", list(conditions))
        self.assertions.append(assertion)
        return assertion

    def assert_eventually(self, *conditions: AttributeCondition) -> BehavioralAssertion:
        """Asserts that the conditions must be true at least once during the run."""
        assertion = BehavioralAssertion("EVENTUALLY", list(conditions))
        self.assertions.append(assertion)
        return assertion
        
# --- DSL Helper to chain conditions (e.g., BQL.EnergyComponent.current_energy > 0) ---

class _ComponentAccessor:
    """Allows accessing attributes on a Component type to build conditions."""
    def __init__(self, component_type: Type[Component]):
        self.component_type = component_type
        self._builder = _build_condition(component_type)
        
    def __getattr__(self, name: str) -> '_AttributeConditionBuilder':
        """Returns a helper object for building the final condition."""
        return _AttributeConditionBuilder(self._builder, name)

class _AttributeConditionBuilder:
    """Provides overloads for comparison operators (>, <, ==) to finalize the condition."""
    def __init__(self, builder: Callable[[str, str, Any], AttributeCondition], attribute_name: str):
        self._builder = builder
        self.attribute_name = attribute_name
        
    def __gt__(self, value: Any) -> AttributeCondition: # Greater Than (>)
        return self._builder(self.attribute_name, ">", value)

    def __lt__(self, value: Any) -> AttributeCondition: # Less Than (<)
        return self._builder(self.attribute_name, "<", value)
        
    def __eq__(self, value: Any) -> AttributeCondition: # Equal To (==)
        return self._builder(self.attribute_name, "=", value)

# --- Example of intended use (for documentation/testing) ---
# Assuming a map containing required components exists:
# component_map = {"TimeBudgetComponent": TimeBudgetComponent, "HealthComponent": HealthComponent}
# bql = BQL(component_map)

# # Example of an assertion:
# assertion1 = bql.assert_always(
#     bql.TimeBudgetComponent.current_time_budget > 0,
#     bql.HealthComponent.current_health > 10.0
# )
# # print(assertion1.to_sql_where_clause())
