# imu_repo/ui_dsl/advanced_components.py
from __future__ import annotations
from typing import Dict, Any, List
import html, json

def _esc(s: str) -> str:
    return html.escape(s, quote=True)

def render_grid(spec: Dict[str,Any], children_html: Dict[str,str]) -> str:
    """
    spec:
      {
        "type": "grid",
        "rows": "auto 1fr auto",
        "cols": "240px 1fr",
        "areas": [
          "sidebar content",
          "sidebar content",
          "footer  footer"
        ],
        "gap": "12px",
        "children": {
          "sidebar": "<div> ... </div>",
          "content": "<div> ... </div>",
          "footer":  "<div> ... </div>"
        }
      }
    """
    rows = spec.get("rows","auto")
    cols = spec.get("cols","1fr")
    gap  = spec.get("gap","8px")
    areas = spec.get("areas") or []
    named = spec.get("children") or {}
    style = [
        "display:grid",
        f"grid-template-rows:{rows}",
        f"grid-template-columns:{cols}",
        f"gap:{gap}",
    ]
    if areas:
        # הופך מערך שורות ל-templateAreas חוקי
        lines = ["\"" + " ".join(r.split()) + "\"" for r in areas]
        style.append(f"grid-template-areas:{' '.join(lines)}")
    grid_children: List[str] = []
    # מרנדר ילד לכל אזור שהוגדר
    for name, inner_html in children_html.items():
        grid_children.append(
            f"<div style='grid-area:{_esc(name)}'>{inner_html}</div>"
        )
    # מפה בשם→HTML: אם חסר מרכיב שהוגדר – מתעלמים בשקט
    for name, inner in (named.items()):
        if name not in children_html and isinstance(inner, str):
            grid_children.append(f"<div style='grid-area:{_esc(name)}'>{inner}</div>")
    return f"<div data-ui='grid' style=\"{';'.join(style)}\">{''.join(grid_children)}</div>"

def _sticky_css(n: int) -> str:
    """
    מחזיר CSS שמקבע n עמודות ראשונות (freeze) בעזרת position:sticky.
    """
    # מייצרים כללים לכל עמודה קפואה: th:nth-child(k), td:nth-child(k) { position:sticky; left:... }
    rules = []
    left = 0
    # רוחב עמודה משוער: משתמשים ב-css var לדיוק אם הוגדר (column-width-k)
    for k in range(1, n+1):
        left_expr = f"var(--col-left-{k}, {left}px)"
        rules.append(f"table[data-freeze] th:nth-child({k}), table[data-freeze] td:nth-child({k}) "
                     f"{{ position: sticky; left: {left_expr}; background: var(--freeze-bg, #fff); z-index:2; }}")
        # משאירים left=0 (התקדמות left מדויקת תיתמך דרך 변수ים דינמיים שמוזרקים ב-JS לאחר מדידה)
    return "<style>" + "\n".join(rules) + "</style>"

def render_table_advanced(spec: Dict[str,Any], rows: List[Dict[str,Any]]) -> str:
    """
    spec:
      {
        "type":"table",
        "columns":[
          {"field":"id","title":"ID","width": "80"},
          {"field":"name","title":"Name"},
          {"field":"score","title":"Score"}
        ],
        "freeze": 1,  # מספר עמודות קפואות מימין לשמאל
        "filters": {"name": {"contains": ""}, "score":{"gte":0}},
        "sort": {"field":"id","dir":"asc"},  # ברירת מחדל
        "search": true
      }
    """
    cols = spec.get("columns") or []
    freeze = int(spec.get("freeze") or 0)
    enable_search = bool(spec.get("search") or False)

    # Header + inputs לסינון
    thead_cells = []
    filter_row_cells = []
    for col in cols:
        title = _esc(col.get("title") or col.get("field") or "")
        width = col.get("width")
        style = f" style='width:{int(width)}px;min-width:{int(width)}px;'" if width else ""
        thead_cells.append(f"<th{style} data-field='{_esc(col.get('field',''))}'>{title}</th>")
        filter_row_cells.append(
            f"<th><input data-filter='{_esc(col.get('field',''))}' placeholder='filter…' /></th>"
        )
    thead = "<thead><tr>" + "".join(thead_cells) + "</tr><tr class='filters'>" + "".join(filter_row_cells) + "</tr></thead>"

    # Body
    body_rows = []
    for r in rows:
        tds = []
        for c in cols:
            field = c.get("field")
            val = r.get(field, "")
            tds.append(f"<td data-field='{_esc(field)}'>{_esc(str(val))}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    table_attrs = "data-ui='table' data-advanced='1'"
    if freeze > 0:
        table_attrs += " data-freeze"
    search_box = "<input id='tbl-search' placeholder='search…' />" if enable_search else ""
    sticky_style = _sticky_css(freeze) if freeze > 0 else ""

    # JS: client-side filter/sort + מדידת עמודות לקיבוע left דינמי
    js = r"""
<script>
(function(){
  const table = document.currentScript.previousElementSibling.querySelector("table");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  const filterInputs = thead.querySelectorAll("tr.filters input[data-filter]");
  const searchBox = document.getElementById("tbl-search");
  const toLower = s => (""+s).toLowerCase();

  function measureFreeze(){
    if(!table.hasAttribute("data-freeze")) return;
    const rows = table.querySelectorAll("tr");
    // מחשבים שמאל מצטבר לעמודות הקפואות מתוך th של השורה הראשונה
    const ths = thead.querySelectorAll("tr:first-child th");
    let left = 0;
    for(let k=0; k<ths.length; k++){
      const th = ths[k];
      const w = th.getBoundingClientRect().width;
      document.documentElement.style.setProperty(`--col-left-${k+1}`, left + "px");
      left += w;
    }
  }

  function applyFilters(){
    const filters = {};
    filterInputs.forEach(inp => {
      const f = inp.getAttribute("data-filter");
      const v = inp.value.trim().toLowerCase();
      if(v.length) filters[f] = v;
    });
    const q = (searchBox && searchBox.value.trim().toLowerCase()) || null;
    const rows = tbody.querySelectorAll("tr");
    rows.forEach(tr => {
      let ok = true;
      if(q){
        ok = toLower(tr.innerText).includes(q);
      }
      if(ok && Object.keys(filters).length){
        for(const [f, v] of Object.entries(filters)){
          const td = tr.querySelector(`td[data-field="${f}"]`);
          const tv = td ? toLower(td.textContent) : "";
          if(!tv.includes(v)){ ok = false; break; }
        }
      }
      tr.style.display = ok ? "" : "none";
    });
  }

  function sortBy(field, dir){
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const getField = (tr) => {
      const td = tr.querySelector(`td[data-field="${field}"]`);
      return td ? td.textContent : "";
    };
    rows.sort((a,b)=>{
      const va = getField(a), vb = getField(b);
      const na = parseFloat(va), nb = parseFloat(vb);
      const bothNum = !isNaN(na) && !isNaN(nb);
      let cmp = 0;
      if(bothNum) cmp = na - nb;
      else cmp = String(va).localeCompare(String(vb));
      return dir==="desc" ? -cmp : cmp;
    });
    rows.forEach(tr => tbody.appendChild(tr));
  }

  // קליקים לכותרות: מיון
  thead.querySelectorAll("tr:first-child th[data-field]").forEach(th=>{
    th.style.cursor = "pointer";
    th.addEventListener("click", ()=>{
      const current = th.getAttribute("data-sort") || "none";
      const next = current==="asc" ? "desc" : "asc";
      thead.querySelectorAll("th[data-field]").forEach(x=>x.removeAttribute("data-sort"));
      th.setAttribute("data-sort", next);
      sortBy(th.getAttribute("data-field"), next);
      measureFreeze();
    });
  });

  // סינון/חיפוש
  filterInputs.forEach(inp=>inp.addEventListener("input", applyFilters));
  if(searchBox) searchBox.addEventListener("input", applyFilters);

  window.addEventListener("resize", measureFreeze);
  setTimeout(()=>{ measureFreeze(); }, 0);
})();
</script>
    """.strip()

    html_table = f"""
<div data-widget='adv-table'>
  {search_box}
  {sticky_style}
  <table {table_attrs}>
    {thead}
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
{js}
    """
    return html_table