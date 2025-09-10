import os
from audit.merkle_tree import merkle_dir
def test_merkle_dir(tmp_path):
    p = tmp_path/"a.txt"; p.write_text("x",encoding="utf-8")
    root, nodes = merkle_dir(str(tmp_path))
    assert isinstance(root,str) and nodes["leaves"]
