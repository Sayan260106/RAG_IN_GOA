def test_grounding_precision():
    context = "Dudhsagar Falls is a four-tiered waterfall located on the Mandovi River in Goa."
    generated = "Dudhsagar Falls is situated on the Mandovi River."
    # Grounding check
    assert "Mandovi River" in context and "Mandovi River" in generated
