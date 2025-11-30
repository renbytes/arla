# tests/agent_core/test_bql.py

import pytest
from typing import Any, Dict, Tuple, List
from agent_core.core.ecs.component import Component
from agent_core.testing import BQL 


# --- Mock Components for Testing ---

class MockEnergyComponent(Component):
    """A mock component to test introspection."""
    current_energy: float
    max_energy: float
    state: str

    def __init__(self, current_energy: float = 100.0):
        self.current_energy = current_energy
        self.max_energy = 100.0
        self.state = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {"current_energy": self.current_energy, "state": self.state}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []

class MockSocialComponent(Component):
    """A mock component to test string handling."""
    partner_id: str
    trust_level: float

    def __init__(self, partner_id: str = "agent_x", trust_level: float = 0.5):
        self.partner_id = partner_id
        self.trust_level = trust_level

    def to_dict(self) -> Dict[str, Any]:
        return {"partner_id": self.partner_id, "trust_level": self.trust_level}
    
    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []

# --- Fixtures ---

@pytest.fixture
def bql():
    """Returns a BQL instance initialized with mock components."""
    # The keys must be the exact names expected by the tests (e.g., EnergyComponent)
    component_map = {
        "MockEnergyComponent": MockEnergyComponent,
        "MockSocialComponent": MockSocialComponent
    }
    # NOTE: BQL's dynamic __init__ routine should now correctly attach these as attributes.
    return BQL(component_map)

# --- Tests ---

def test_bql_setup_and_assertion(bql):
    """
    Tests that the basic assertion structure can be created 
    without immediate execution errors.
    """
    assertion = bql.assert_always(
        bql.MockEnergyComponent.current_energy > 0.0,
    )
    assert assertion.temporal == "ALWAYS"
    assert len(assertion.conditions) == 1


def test_introspection(bql):
    """Ensure BQL correctly discovers attributes via the Component._get_component_fields method."""
    # Accessors are attached via the BQL instance
    assert bql.MockEnergyComponent.current_energy
    assert bql.MockSocialComponent.partner_id

    # Should raise error for non-existent attribute
    with pytest.raises(ValueError) as excinfo:
        _ = bql.MockEnergyComponent.non_existent_field > 0
    assert "Attribute 'non_existent_field' not found" in str(excinfo.value)

def test_numeric_assertion_compilation(bql):
    """Test compilation of numeric comparisons (>, <, =)."""
    # Assertion: Energy > 0
    condition = bql.MockEnergyComponent.current_energy > 0
    sql_fragment = str(condition)
    
    # Expected: Cast to FLOAT for numeric comparison
    expected = "(components_data->'MockEnergyComponent'->>'current_energy')::FLOAT > 0"
    assert sql_fragment == expected

def test_string_assertion_compilation(bql):
    """Test compilation of string comparisons."""
    # Assertion: State == 'active'
    condition = bql.MockEnergyComponent.state == "active"
    sql_fragment = str(condition)
    
    # Expected: Cast to TEXT (implicit in JSONB->>) and quoted value
    expected = "(components_data->'MockEnergyComponent'->>'state')::TEXT = 'active'"
    assert sql_fragment == expected

def test_complex_always_assertion(bql):
    """Test a multi-condition ALWAYS assertion."""
    assertion = bql.assert_always(
        bql.MockEnergyComponent.current_energy > 0,
        bql.MockEnergyComponent.state == "active"
    )

    assert assertion.temporal == "ALWAYS"
    where_clause = assertion.to_sql_where_clause()
    
    expected_part_1 = "(components_data->'MockEnergyComponent'->>'current_energy')::FLOAT > 0"
    expected_part_2 = "(components_data->'MockEnergyComponent'->>'state')::TEXT = 'active'"
    
    assert expected_part_1 in where_clause
    assert " AND " in where_clause
    assert expected_part_2 in where_clause

def test_eventually_assertion(bql):
    """Test an EVENTUALLY assertion."""
    assertion = bql.assert_eventually(
        bql.MockSocialComponent.trust_level > 0.9
    )
    
    assert assertion.temporal == "EVENTUALLY"
    where_clause = assertion.to_sql_where_clause()
    # Corrected component name in expected output
    expected = "(components_data->'MockSocialComponent'->>'trust_level')::FLOAT > 0.9"
    assert where_clause == expected