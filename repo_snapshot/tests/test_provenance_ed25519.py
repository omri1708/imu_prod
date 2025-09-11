# tests/test_provenance_ed25519.py
from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record, verify_envelope

def test_keyring_rotate_and_sign_verify(tmp_path):
    kr=Keyring(str(tmp_path/"keys"))
    meta=kr.rotate("test")
    assert kr.current_kid()==meta.kid
    priv=kr.load_private(meta.kid)
    pub=kr.load_public(meta.kid)
    rec={"digest":"deadbeef"*8,"kind":"artifact","meta":{"k":"v"}}
    env=sign_cas_record(priv, meta.kid, rec)
    assert verify_envelope(pub, env) is True