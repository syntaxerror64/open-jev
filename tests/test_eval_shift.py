from eval.shift import new_keys, paraphrase, shuffle_keys, structure_noise

def test_shuffle_keys_keeps_values() -> None:
    s = {"a": 1, "b": 2, "c": 3}
    out = shuffle_keys(s, seed=0)
    assert sorted(map(str, out.values())) == sorted(map(str, s.values()))
    assert set(out) == set(s)

def test_structure_noise_changes_shape_and_is_deterministic() -> None:
    s = {"a": 1, "b": 2}
    n1, n2 = structure_noise(s, seed=7), structure_noise(s, seed=7)
    assert n1 == n2                              # фиксированный seed → одинаково
    assert set(n1) != set(s)                     # состав ключей изменился

def test_new_keys_are_actually_new() -> None:
    s = {"a": 1}
    out = new_keys(s, n=3, seed=1)
    assert set(out) - set(s) and not (set(s) - set(out))

def test_paraphrase_preserves_keys_and_leaves_values_touchable() -> None:
    s = {"customer": {"tier": "enterprise"}}
    out = paraphrase(s, seed=0)
    assert set(out) == set(s)                    # переформулировка не ломает структуру
