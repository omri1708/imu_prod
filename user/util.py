# אופציונלי: גרסת בדיקה אם בכל זאת רוצים לשמור utility
from ui.dsl import Page, Component
from ui.schema_compose import apply_table_specs

def finalize_with_auto_remediation_for_specs(
    table_specs: List[Dict[str,Any]],
    *,
    policy: Dict[str,Any],
    runtime_fetcher=None,
    attempts: int = 3,
) -> Dict[str,Any]:
    # מרכיבים Page זמני לצורך הגארד
    page = Page(title="temporary")
    # משקפים את ה-specs אל ה-Page (כדי שהגארד ידע לקרוא אותם)
    apply_table_specs(page, table_specs, mode="merge")
    evs = []  # או הזרק evidences מתאימים אם יש

    last = None
    for i in range(1, attempts+1):
        try:
            out = run_negative_suite(page, evs, policy=policy, runtime_fetcher=runtime_fetcher)
            record_event("finalize_ok", {"attempt": i}, severity="info")
            return {"ok": True, "attempt": i, "guard": out}
        except RolloutBlocked as rb:
            last = rb
            record_event("finalize_blocked", {"attempt": i, "reason": str(rb)}, severity="warn")
            diags = diagnose(rb.__cause__ or rb)
            rems = propose_remedies(diags, policy=policy, table_specs=table_specs)
            if not rems or i == attempts:
                break
            apply_remedies(rems, policy=policy, table_specs=table_specs)
            apply_table_specs(page, table_specs, mode="merge")
    raise last if last else RuntimeError("finalize failed")
