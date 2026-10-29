from canal.chem import nm_to_p_activity


def test_nm_to_p_activity_exact():
    value = nm_to_p_activity(1000, "=")
    assert value is not None
    assert abs(value.p_activity - 6.0) < 1e-8
    assert value.p_relation == "="


def test_nm_relation_reverses_for_pactivity_lower_activity():
    value = nm_to_p_activity(1000, ">")
    assert value is not None
    assert abs(value.p_activity - 6.0) < 1e-8
    assert value.p_relation == "<"


def test_nm_relation_reverses_for_pactivity_higher_activity():
    value = nm_to_p_activity(1000, "<=")
    assert value is not None
    assert abs(value.p_activity - 6.0) < 1e-8
    assert value.p_relation == ">"
