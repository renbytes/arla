from simulations.schelling_sim.components import PositionComponent


class TestPositionComponent:
    def test_init(self):
        comp = PositionComponent(x=10, y=20)
        assert comp.x == 10
        assert comp.y == 20
        assert comp.previous_x == 10
        assert comp.previous_y == 20

    def test_move_to(self):
        comp = PositionComponent(x=10, y=20)
        comp.move_to(new_x=15, new_y=25)
        assert comp.x == 15
        assert comp.y == 25
        assert comp.previous_x == 10
        assert comp.previous_y == 20

    def test_position_property(self):
        comp = PositionComponent(x=10, y=20)
        assert comp.position == (10, 20)

    def test_to_dict(self):
        comp = PositionComponent(x=10, y=20)
        expected_dict = {"x": 10, "y": 20, "previous_x": 10, "previous_y": 20}
        assert comp.to_dict() == expected_dict

    def test_validate_success(self):
        valid_comp = PositionComponent(x=10, y=20)
        is_valid, errors = valid_comp.validate("agent_1")
        assert is_valid
        assert not errors

    def test_validate_failure(self):
        invalid_comp = PositionComponent(x=10, y=20.5)  # Invalid type
        is_valid, errors = invalid_comp.validate("agent_1")
        assert not is_valid
        assert "Position coordinates must be integers" in errors
