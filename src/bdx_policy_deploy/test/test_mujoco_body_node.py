from bdx_policy_deploy.mujoco_body_node import effective_mass_profile, is_leg_mass_body_name


def test_leg_mass_profile_includes_hip_supports() -> None:
    assert is_leg_mass_body_name("Left_Hip_Yaw_Support")
    assert is_leg_mass_body_name("Right_Hip_Yaw_Support")
    assert is_leg_mass_body_name("Left_Hip_Yaw_Motor")
    assert is_leg_mass_body_name("Right_Foot_Cover")


def test_leg_mass_profile_excludes_non_leg_body_names() -> None:
    assert not is_leg_mass_body_name("base_link")
    assert not is_leg_mass_body_name("Battery")
    assert not is_leg_mass_body_name("Head_Yaw_Motor")
    assert not is_leg_mass_body_name("Left_Holder")


def test_policy_only_mass_profile_keeps_full_mass_until_policy_mode() -> None:
    assert effective_mass_profile("disabled", "legs_only", "policy_only") == "full"
    assert effective_mass_profile("zero_action", "legs_only", "policy_only") == "full"
    assert effective_mass_profile("policy", "legs_only", "policy_only") == "legs_only"
    assert effective_mass_profile("disabled", "legs_only", "always") == "legs_only"


def test_enabled_only_mass_profile_applies_outside_disabled_mode() -> None:
    assert effective_mass_profile("disabled", "legs_only", "enabled_only") == "full"
    assert effective_mass_profile("zero_action", "legs_only", "enabled_only") == "legs_only"
    assert effective_mass_profile("policy", "legs_only", "enabled_only") == "legs_only"
