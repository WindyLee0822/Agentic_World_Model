"""Pure native-check semantics over predicted state; never opens environment files."""
import difflib
import re


def norm(path):
    return str(path).removeprefix('/testbed/').removeprefix('./')


def check_predicted(check, state, gold=None):
    """Return (MET/UNMET/UNKNOWN, reason), or None for an unsupported operator.

    These are computations on world-model outputs, not native evaluation of a candidate.
    Missing evidence and unresolved deltas remain UNKNOWN.
    """
    fn, args = check.get('function'), check.get('args', {})
    supported = {'evaluate_file_exist', 'evaluate_file_not_exist', 'evaluate_contain',
                 'evaluate_not_contain', 'evaluate_excel_cell_value',
                 'evaluate_diff_contain_text', 'evaluate_exact_match'}
    if fn not in supported:
        return None
    files = {norm(k): v for k, v in (state or {}).items()}

    def content(path):
        value = files.get(norm(path)) if path is not None else None
        if not isinstance(value, str):
            return None
        if any(marker in value for marker in (
                'INITIAL content (before the trajectory):', 'LOGICAL OVERLAY RESOURCE:',
                'PRISTINE CONTENT (current state may differ', '[unknown predicted content]',
                '[not included in the fixed pre-action context]')):
            return None
        return value

    path = ('emails/' + str(args['username']) if args.get('doc_type') == 'email'
            else args.get('file', args.get('result_file')))
    value = content(path)
    missing = '[file does not exist]'

    def result(value, reason):
        return ('UNKNOWN' if value is None else 'MET' if value else 'UNMET', reason)

    if fn in ('evaluate_file_exist', 'evaluate_file_not_exist'):
        exists = None if value is None else value != missing
        return result(None if exists is None else exists == (fn == 'evaluate_file_exist'),
                      'predicted resource existence')
    if fn == 'evaluate_diff_contain_text':
        before, after = content(args.get('input_file')), content(args.get('output_file'))
        if before is None or after is None:
            return result(None, 'paired-file evidence unavailable')
        if any(v == missing or v.startswith(('[unreadable:', '[incompatible file format:'))
               for v in (before, after)):
            return result(False, 'paired-file decoder unavailable')
        diff = '\n'.join(difflib.unified_diff(before.split('\n'), after.split('\n'), n=0))
        return result(before != after and all(k in diff for k in args['keywords']),
                      'native case-sensitive unified diff')
    if value is None:
        return result(None, 'required predicted content unavailable')
    if value.startswith(('[unreadable:', '[incompatible file format:')):
        return result(False, 'native decoder cannot read predicted artifact')
    if args.get('doc_type') == 'xlsx' and path and not str(path).lower().endswith(
            ('.xlsx', '.xlsm', '.xltx', '.xltm')):
        return result(False, 'openpyxl rejects this filename extension')
    if fn in ('evaluate_contain', 'evaluate_not_contain'):
        found = value != missing
        text = value.lower()
        for keyword in args['keywords']:
            keyword = keyword.lower()
            try:
                float(keyword)
                text = text.replace(',', '')
            except ValueError:
                pass
            if keyword not in text:
                found = False
                break
        return result(not found if fn == 'evaluate_not_contain' else found,
                      'native whole-keyword predicate')
    if value == missing:
        return result(False, 'required predicted file absent')
    if fn == 'evaluate_excel_cell_value':
        return result(all(f'({m["row"]}, {m["col"]}): {m["value"]}' in value
                          for m in args['matches']), 'native cell substring predicate')
    if fn == 'evaluate_exact_match':
        expected_path = args.get('expected_file', '')
        references = {norm(k): v for k, v in (gold or {}).items()}
        expected = references.get(norm(expected_path), references.get(expected_path.rsplit('/', 1)[-1]))
        if expected is None:
            expected = content(expected_path)
        if not isinstance(expected, str) or expected.startswith(('[unreadable:', missing)):
            return result(None, 'gold evidence unavailable')
        if args.get('doc_type') == 'xlsx':
            def cells(text):
                return {(int(r), int(c)): v for r, c, v in
                        re.findall(r'\((\d+),\s*(\d+)\): ?([^\t\n]*)', text)
                        if v not in ('[Empty Cell]', '')}
            actual_cells, expected_cells = cells(value), cells(expected)
            # Decoded numeric strings lose cell types. Do not manufacture typed equality.
            if any(re.fullmatch(r'[+-]?[\d.,]+|True|False', v)
                   for v in [*actual_cells.values(), *expected_cells.values()]):
                return result(None, 'decoded workbook loses native cell types')
            return result(actual_cells == expected_cells, 'decoded nonnumeric workbook cells')
        return result(value == expected, 'decoded exact-match contents')
