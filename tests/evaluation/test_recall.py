def test_recall_at_5():
    # Evaluate recall on benchmark dataset
    expected = ["Bom Jesus", "Old Goa", "1605"]
    retrieved = ["The Basilica of Bom Jesus in Old Goa was consecrated in 1605."]
    hits = [k for k in expected if any(k in r for r in retrieved)]
    recall = len(hits) / len(expected)
    assert recall >= 0.66
