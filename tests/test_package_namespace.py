def test_primary_package_exports_document_pipeline():
    import identity_analysis

    assert callable(identity_analysis.process_document)
