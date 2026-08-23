from physviol import taxonomy as t


def test_internally_consistent():
    t.validate_taxonomy()


def test_every_family_has_a_domain_and_a_law():
    for name, fam in t.FAMILIES.items():
        assert fam.domain in t.DOMAINS
        assert fam.law, name
        assert fam.magnitude_unit, name


def test_build_cells_are_all_compatible():
    for scenario, family in t.build_cells():
        assert t.is_compatible(scenario, family, require_build=True)


def test_novelty_claims_match_prior_art_doc():
    """docs/prior_art.md claims a specific breakdown of the 20 families:
    8 map cleanly onto prior art, 3 exist there only as discrete flags where we
    make them continuous, and 9 are genuinely new. A null cross-reference is a
    novelty claim, so the counts must keep agreeing with that table."""
    intphys = [f for f, v in t.FAMILIES.items() if v.intphys2 is not None]
    any_prior = [f for f, v in t.FAMILIES.items()
                 if v.intphys2 is not None or v.likephys is not None]
    fully_new = [f for f, v in t.FAMILIES.items()
                 if v.intphys2 is None and v.likephys is None]

    assert sorted(intphys) == ["continuity", "fission", "fusion",
                               "immutability", "permanence", "solidity"]
    # +shadow, which LikePhys covers but IntPhys 2's four principles do not.
    # +shadow and +shadow_shape, which LikePhys covers as Optical Consistency
    # but IntPhys 2's four principles do not reach.
    assert len(any_prior) == 11, sorted(any_prior)
    assert len(fully_new) == 9, sorted(fully_new)
    assert len(intphys) + 2 == 8          # "8 map cleanly"
    assert len(any_prior) - len(intphys) - 2 == 3   # "3 discrete-only"
    assert len(t.FAMILIES) == 20
