"""Leakage and integrity tests requested in review (Academia comments set 2, item 3)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcare.data import load_tables, verify_checksums  # noqa: E402
from mcare.experiment import leakage_groups, make_splits  # noqa: E402
from mcare.frames import build_frames  # noqa: E402
from mcare.graph import build_fold_graph  # noqa: E402
from mcare.models import GraphSAGEClassifier, set_seed  # noqa: E402
from mcare.preprocess import clean_for_vectoriser  # noqa: E402
from mcare.taxonomy import CATEGORIES  # noqa: E402


@pytest.fixture(scope="module")
def data():
    t = load_tables()
    mapping, q, a = build_frames(t, write=False)
    y = q.category.map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy()
    clean = [clean_for_vectoriser(x) for x in q.text]
    return t, mapping, q, y, clean


def test_checksums_match_dataverse():
    assert all(v["ok"] for v in verify_checksums().values())


def test_every_intent_has_exactly_one_category(data):
    t, mapping, *_ = data
    assert len(mapping) == len(t.intents) == 503
    assert mapping.category.isin(CATEGORIES).all()
    assert mapping.intent_id.is_unique


def test_groups_never_straddle_train_and_test(data):
    _, _, q, y, _ = data
    sp = make_splits(y, q.group.to_numpy(), q.row_id.to_numpy())
    g = dict(zip(q.row_id, q.group))
    sp["group"] = sp.row_id.map(g)
    for (_, _), s in sp.groupby(["repeat", "fold"]):
        roles = s.groupby("group").role.nunique()
        assert (roles == 1).all()
    # each sample is tested exactly once per repetition
    te = sp[sp.role == "test"].groupby("repeat").row_id.apply(lambda s: s.is_unique and len(s) == len(q))
    assert te.all()


def test_duplicate_texts_share_a_group(data):
    t, mapping, *_ = data
    groups = leakage_groups(t.intents, t.patterns, t.responses)
    dup = mapping[mapping.duplicated("tag", keep=False)]
    for _, d in dup.groupby("tag"):
        assert len({groups[i] for i in d.intent_id}) == 1


def test_graph_has_no_label_nodes_and_heldout_nodes_send_no_messages(data):
    _, _, q, y, clean = data
    n = len(y)
    train = np.zeros(n, bool)
    train[: int(0.8 * n)] = True
    g = build_fold_graph(clean, np.zeros(n), train)
    assert g.x.shape[0] == n + g.n_terms          # only question + term nodes
    src, dst = g.edge_index.numpy()
    q_src = src[src < n]
    assert train[q_src].all()                      # only training questions send messages
    assert (dst[src < n] >= n).all()               # question -> term only
    assert (src[dst < n] >= n).all()               # questions receive only from terms


def test_heldout_prediction_invariant_to_other_heldout_texts(data):
    """Inductive check: altering the text of one held-out question must not
    change predictions for any other held-out question."""
    _, _, q, y, clean = data
    n = len(y)
    train = np.zeros(n, bool)
    train[: int(0.8 * n)] = True
    g1 = build_fold_graph(clean, np.zeros(n), train)
    clean2 = list(clean)
    clean2[n - 1] = "emergency bleeding fever"     # perturb one held-out question
    g2 = build_fold_graph(clean2, np.zeros(n), train)
    set_seed(0)
    m = GraphSAGEClassifier(g1.x.shape[1], 64, len(CATEGORIES)).eval()
    with torch.no_grad():
        p1 = m(g1.x, g1.edge_index)[: n - 1]
        p2 = m(g2.x, g2.edge_index)[: n - 1]
    assert torch.allclose(p1, p2, atol=1e-5)


def test_labels_are_not_inputs_to_graph_builder():
    import inspect
    params = inspect.signature(build_fold_graph).parameters
    assert not any(p in params for p in ("y", "labels", "category", "intent"))


def test_no_hardcoded_results_in_package():
    """The previous release returned Table 6 as literals; guard against regressions."""
    src = "".join(p.read_text() for p in (Path(__file__).resolve().parents[1] / "mcare").glob("*.py"))
    for literal in ("0.914", "0.889", "0.956", "Med-PaLM", "Llama-3-70B", "np.random.beta", "np.random.uniform"):
        assert literal not in src
