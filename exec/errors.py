# imu_repo/exec/errors.py


class ExecError(Exception): ...


class ResourceRequired(Exception):
    def __init__(self, what: str, how: str):
        super().__init__(f"resource_required:{what}")
        self.what=what; self.how=how