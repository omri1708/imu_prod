# imu_repo/ui/forms.py
from __future__ import annotations
import json, html
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Rule:
    field: str
    type: str                   # "string" | "number" | "boolean"
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

@dataclass
class FormSchema:
    rules: List[Rule] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FormSchema":
        rules = []
        for r in d.get("rules", []):
            rules.append(Rule(**r))
        return FormSchema(rules=rules)

def _esc_json(d: Dict[str,Any]) -> str:
    return html.escape(json.dumps(d, separators=(",",":")), quote=True)

def compile_schema_to_js(schema: FormSchema) -> str:
    """
    מחזיר מחרוזת JS (פונקציה טהורה) שמקבלת אובייקט formData ומחזירה {ok, errors}
    ללא תלות חיצונית.
    """
    obj = {
        "rules":[r.__dict__ for r in schema.rules]
    }
    payload = _esc_json(obj)
    return f"""
(function(){{
  const schema = {payload};
  function typeOf(v){{
    if (typeof v === 'boolean') return 'boolean';
    if (typeof v === 'number' && isFinite(v)) return 'number';
    if (typeof v === 'string') return 'string';
    return 'unknown';
  }}
  return function validate(formData){{
    const errors = [];
    for (const rule of schema.rules){{
      const v = formData[rule.field];
      if (rule.required && (v === undefined || v === null || v === '')){{
        errors.push({{field: rule.field, msg: 'required'}});
        continue;
      }}
      if (v === undefined || v === null || v === '') continue;
      const t = typeOf(v);
      if (rule.type && t !== rule.type){{
        // נסה להמיר מספר ממחרוזת
        if (rule.type === 'number' && typeof v === 'string'){{
          const n = Number(v);
          if (!Number.isFinite(n)) {{
            errors.push({{field: rule.field, msg: 'type'}}); 
            continue;
          }} else {{
            formData[rule.field] = n;
          }}
        }} else if (rule.type === 'boolean' && typeof v === 'string'){{
          const b = (v === 'true' || v === '1');
          formData[rule.field] = b;
        }} else {{
          errors.push({{field: rule.field, msg: 'type'}});
          continue;
        }}
      }}
      if (rule.min_length != null && String(v).length < rule.min_length) {{
        errors.push({{field: rule.field, msg: 'min_length'}});
      }}
      if (rule.max_length != null && String(v).length > rule.max_length) {{
        errors.push({{field: rule.field, msg: 'max_length'}});
      }}
      if (rule.pattern){{
        try {{
          const re = new RegExp(rule.pattern);
          if (!re.test(String(v))) errors.push({{field: rule.field, msg:'pattern'}});
        }} catch (e) {{ errors.push({{field: rule.field, msg:'bad_pattern'}}); }}
      }}
      if (rule.minimum != null && Number(v) < rule.minimum) {{
        errors.push({{field: rule.field, msg:'minimum'}});
      }}
      if (rule.maximum != null && Number(v) > rule.maximum) {{
        errors.push({{field: rule.field, msg:'maximum'}});
      }}
    }}
    return {{ ok: errors.length === 0, errors }};
  }}
}})()
""".strip()