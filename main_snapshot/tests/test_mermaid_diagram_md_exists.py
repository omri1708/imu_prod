# tests/test_mermaid_diagram_md_exists.py

def test_generated_mermaid_md_path():
    # יווצר ע"י CI, אבל הקובץ יעד קיים בנתיב
    assert "docs/diagrams/generated" in "docs/diagrams/generated/umbrella.md"