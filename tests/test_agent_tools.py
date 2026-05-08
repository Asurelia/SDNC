from sdnc.agent.tools import CalculatorTool


def test_calculator_extracts_symbolic_expression():
    result = CalculatorTool().run("calcule 2 + 2", {})

    assert result.success
    assert result.content == "2 + 2 = 4"


def test_calculator_extracts_simple_natural_subtraction():
    result = CalculatorTool().run("Si j'ai 3 pommes et que j'en donne 1, combien il m'en reste ?", {})

    assert result.success
    assert result.content == "3 - 1 = 2"


def test_calculator_rejects_number_lists_without_operation():
    result = CalculatorTool().run("les codes sont 12 34 56", {})

    assert not result.success
