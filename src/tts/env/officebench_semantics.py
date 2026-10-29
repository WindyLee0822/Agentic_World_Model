"""Pure OfficeBench representation and CLI semantics; never executes an app action."""
from __future__ import annotations

import ast
import re
import shlex


# Public application semantics only. Never interpolate task file contents into these contracts.
_WRITE_TEMPLATES = {
    ('excel', 'set_cell'): ('Successfully write text to {file_path}', 'Failed to write text to {file_path}', 'The file {file_path} does not exist. Failed to write to the file.'),
    ('excel', 'delete_cell'): ('Successfully delete a cell in {file_path}', 'Failed to delete a cell in {file_path}', 'The file {file_path} does not exist. Failed to delete the cell in the file.'),
    ('word', 'write_to_file'): ('Successfully write contents to {file_path}', 'Failed to write contents to {file_path}', 'The file {file_path} does not exist. Failed to write to the file.'),
}


def email_cli_data(action):
    """Native Fire recipient semantics, without invoking the app or a shell."""
    try:
        value = ('Multiple recipients' if isinstance(action.get('recipient'), list)
                 else cli_literal(action.get('recipient', '')))
    except (ValueError, SyntaxError):
        return {'recipient_display': str(action.get('recipient', '')), 'recipient_kind': 'unparseable',
                'multiple_recipients': False, 'recipient': action.get('recipient', ''), 'parse_error': True}
    return {'recipient_display': str(value), 'recipient_kind': type(value).__name__,
            'multiple_recipients': value == 'Multiple recipients',
            'recipient': value}


def action_contract(action):
    """Data-independent requirements for the generated transition/rendering program."""
    key = (action.get('app'), action.get('action'))
    common = (
        "CURRENT STATE CONTRACT: data['current_state'] maps resource paths to records. "
        "Keys use paths relative to /testbed, exactly as in data['targets'] (writes) or "
        "data['target'] (reads), NOT absolute action paths. For a write, retrieve "
        "record = data['current_state'].get(data['targets'][0], {}). The record fields are "
        "status ('exact' or 'unknown'), exists (True/False/None), and content (str/None). "
        "Presence of a record/key does NOT mean the file exists; inspect record['exists']. "
        "An exact record's content is already AFTER applied writes; never apply writes twice. "
        "An unknown record is NOT evidence of absence: inspect its pristine_content and ordered "
        "transitions. A conversion_source record contains the copied logical input state AT "
        "conversion time; use it to predict decoded output text without running a converter. "
        "Rejected transitions never change state. READ results are pristine. "
        "Use data['observation_action'] for response arguments and data['action'] for resolved paths. "
        "Never replace an observation with paraphrased wording or inferred file contents.\n"
    )
    if key in _WRITE_TEMPLATES:
        return common + (
            "WRITE RESPONSE CONTRACT: choose exactly one of these templates after checking current "
            "existence, input types, and the app preconditions; interpolate action arguments in Python. "
            "Templates (success, operation failure, missing file): " + repr(_WRITE_TEMPLATES[key])
        )
    if key[0] in ('word', 'excel') and key[1] == 'convert_to_pdf':
        return common + (
            "CONVERSION CONTRACT: current_state includes INPUTS as well as outputs; check the "
            "input record, not pristine absence or the output's existence. Linux LibreOffice uses "
            "the input basename with .pdf in dirname(pdf_file_path), ignoring the requested output "
            "basename. The wrapper prints 'Successfully convert {input} to {pdf_file_path}' after "
            "launching LibreOffice even if its exit code is nonzero; that wording alone does not "
            "prove an artifact exists. A missing Word input prints 'The word file {input} does not "
            "exist. Failed to convert the file to pdf.' A missing Excel input prints '{input} does "
            "not exist. Failed to convert {input} to {pdf_file_path}'. Substitute raw action paths. "
            "Word create/write produces DOCX even with a .pdf suffix; it is not a PDF conversion."
        )
    if key == ('email', 'send_email'):
        return common + (
            "EMAIL CONTRACT: sender and recipient are explicit identities, with no hidden user ID. "
            "A list recipient becomes 'Multiple recipients' and returns 'Failed to send email to "
            "Multiple recipients. Only support one recipient.'. Otherwise success is 'Successfully "
            "sent email to {recipient}.' and failure is 'Failed to send email to {recipient}.'. "
            "Use data['email_cli']['recipient_display'] for the recipient AFTER Fire parsing, "
            "including @domain in the response. Strip @domain only "
            "for mailbox paths. Native send creates BOTH sender and recipient directories, writes "
            "the same {subject}.eml in both, and overwrites an existing identical subject. It strips "
            "single/double quotes from subject/content before Fire argument parsing. It does not "
            "attach files or require a mailbox to exist. Fire parses unquoted 'Bob, Tom' as a "
            "TUPLE: sending fails and the response displays ('Bob', 'Tom'). When recipient_kind "
            "is tuple or list, use the failure template, never success. A literal list action "
            "argument instead becomes the special 'Multiple recipients' rejection above."
        )
    if key == ('ocr', 'recognize_file'):
        return common + (
            "OCR CONTRACT: data['reads'][0]['result'] is already a STRING of OCR text, not a "
            "dictionary or bytes. Preserve it verbatim and prefix 'The text from {file_path} is:\\n'. "
            "Missing input: 'The file {file_path} does not exist. Failed to recognize text.'. "
            "Empty text or [OCR failed]: 'Failed to recognize text from {file_path}'. Trim only "
            "the complete driver-visible result."
        )
    if key in (('excel', 'read_file'), ('word', 'read_file')):
        header = ('The following is the table from the excel file:\n' if key[0] == 'excel'
                  else 'The following is the content from the word file:')
        return common + (
            "READ RESPONSE CONTRACT: the copied operand is decoded content, not the app response. "
            f"Prepend the native header {header!r}. For Word append newline before each paragraph; "
            "for Excel preserve every cell, tab, and row separator. A missing file returns "
            "'The file {file_path} does not exist. Failed to read the file.'. Trim outer whitespace "
            "only after constructing the complete driver-visible observation."
        )
    if key == ('calendar', 'list_events'):
        return common + (
            "CALENDAR RENDERING CONTRACT: username selects /testbed/calendar/{username}.ics; "
            "there is no separate authenticated user ID. Parse ICS, unfold continuation lines, "
            "Use only allowed re/datetime/string operations; dateutil and icalendar imports are "
            "unavailable in the restricted worker. datetime.strptime and strftime are allowed. "
            "unescape TEXT values (\\\\, \\n, \\, and \\;), and preserve VEVENT order. "
            "Do not print raw DTSTART/DTEND strings. Native format_time calls "
            "obj.dt.strftime('%Y-%m-%d %H:%M:%S'): convert YYYYMMDDTHHMMSS[Z] to "
            "YYYY-MM-DD HH:MM:SS; DATE values YYYYMMDD render at 00:00:00. "
            "TZID values retain their local wall-clock time; do not convert to UTC. "
            "Each event renders Summary, Start Time, End Time, Description, Location, each as "
            "'Field: value' on its own line, followed by exactly 50 hyphens. Missing properties "
            "render None. Join event blocks with newline and trim the final result. "
            "For nonempty events prepend 'Successfully list events for {username}:\\n'. "
            "An empty calendar returns 'No events found for {username}.'. "
            "A parse failure returns 'Failed to list events for {username}.'. "
            "A missing calendar is created empty by native list_events if its parent exists; "
            "a missing parent causes an exception, not an empty-success observation. "
            "Generate code using data values, never literal event values."
        )
    return common


def observation_contract_error(context, observation):
    """Reject invented write wording without exposing any data in repair feedback."""
    if not context.get('enforce_observation_contract'):
        return None
    if context.get('kind') == 'final_file_state' and context.get('conversion_source'):
        state = context['conversion_source'].get('state', {})
        if state.get('status') == 'exact' and state.get('exists') is True and str(observation).strip() == '[file does not exist]':
            return ('Pristine output absence is not proof of absence after the predicted conversion. '
                    "Inspect data['conversion_source']['state']['content']. If conversion output "
                    'cannot be predicted, report [unknown predicted content] rather than inventing absence.')
        return None
    action = context.get('observation_action', context.get('action', {}))
    rendered = str(observation).split('OBSERVATION:')[-1].strip()
    key = (action.get('app'), action.get('action'))
    if key[0] in ('word', 'excel') and key[1] == 'convert_to_pdf':
        source_key = key[0] + '_file_path'
        source = action.get(source_key)
        output = action.get('pdf_file_path')
        missing = (f'The word file {source} does not exist. Failed to convert the file to pdf.'
                   if key[0] == 'word' else f'{source} does not exist. Failed to convert {source} to {output}')
        inputs = context.get('inputs') or []
        record = context.get('current_state', {}).get(inputs[0], {}) if inputs else {}
        if record.get('status') == 'exact':
            if record.get('exists') is False and rendered != missing:
                return 'The current INPUT is exactly missing; use the native missing-input template.'
            if record.get('exists') is True and 'does not exist' in rendered:
                return 'The current INPUT exists after predicted writes; pristine absence is obsolete.'
        return None
    if key == ('email', 'send_email'):
        cli = email_cli_data(action)
        if cli.get('parse_error'):
            return None
        recipient = cli['recipient_display']
        if cli['multiple_recipients']:
            allowed = ['Failed to send email to Multiple recipients. Only support one recipient.']
        elif cli['recipient_kind'] in ('tuple', 'list'):
            allowed = [f'Failed to send email to {recipient}.']
        else:
            allowed = [f'Successfully sent email to {recipient}.', f'Failed to send email to {recipient}.']
        if rendered not in allowed:
            return 'Use the exact native email response template with the Fire-parsed recipient display and type; tuples/lists cannot succeed.'
        return None
    if key == ('ocr', 'recognize_file') and context.get('reads'):
        operand = context['reads'][0]
        path = action.get('file_path')
        content = operand['result']
        if operand.get('status') == 'missing':
            expected = f'The file {path} does not exist. Failed to recognize text.'
        elif not content or content == '[OCR failed]':
            expected = f'Failed to recognize text from {path}'
        else:
            expected = f'The text from {path} is:\n{content}'
        if rendered != expected.split('OBSERVATION:')[-1].strip():
            return 'OCR output must preserve the copied string exactly with the documented native wrapper.'
        return None
    if key in (('excel', 'read_file'), ('word', 'read_file')) and context.get('kind') == 'clean_read':
        # Compare decoded evidence to the native wrapper without making an app call or
        # returning private contents in the repair prompt.
        reads = context.get('reads') or []
        if reads and reads[0].get('status') in ('ok', 'missing'):
            content = reads[0]['result']
            if reads[0]['status'] == 'missing':
                expected = f"The file {action.get('file_path')} does not exist. Failed to read the file."
            elif key[0] == 'excel':
                expected = 'The following is the table from the excel file:\n' + content
            else:
                expected = 'The following is the content from the word file:' + ('\n' + content if content else '')
            if rendered != expected.split('OBSERVATION:')[-1].strip():
                return 'Read observation differs from the copied operand and native wrapper. Preserve all content and apply only the documented rendering.'
        return None
    if (action.get('app'), action.get('action')) == ('calendar', 'list_events'):
        username = action.get('username')
        if not isinstance(username, str):
            return None
        if rendered in (f'No events found for {username}.', f'Failed to list events for {username}.'):
            return None
        if not rendered.startswith(f'Successfully list events for {username}:\n'):
            return 'Calendar observation must use the native success header or empty/failure template.'
        for line in rendered.splitlines():
            if line.startswith(('Start Time:', 'End Time:')):
                value = line.split(': ', 1)[-1]
                if value != 'None' and not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', value):
                    return "Convert ICS timestamps with native strftime('%Y-%m-%d %H:%M:%S'); raw ICS timestamps are invalid."
        return None
    templates = _WRITE_TEMPLATES.get((action.get('app'), action.get('action')))
    if templates is None:
        return None
    try:
        allowed = [template.format(**action) for template in templates]
    except (KeyError, ValueError):
        return None
    if rendered not in allowed:
        return 'Observation violates the native response templates. Use the action contract; do not paraphrase.'
    # Exact absence is authoritative; unknown state is deliberately not classified here.
    target = context.get('targets', [])
    record = context.get('current_state', {}).get(target[0], {}) if target else {}
    if record.get('status') == 'exact':
        if record.get('exists') is False and rendered != allowed[2]:
            return 'Observation contradicts exact current-state absence. Use the missing-file template.'
        if record.get('exists') is True and rendered == allowed[2]:
            return 'Observation contradicts exact current-state existence. Do not use pristine absence as current absence.'
    return None

NATIVE_SEMANTICS = """
OfficeBench implementation rules (not generic office-software behavior):
- Predict the DRIVER-visible observation, not raw app stdout: remove the final
  'OBSERVATION:' prefix and trim leading/trailing whitespace. Excel read_file starts
  'The following is the table from the excel file:\n'; Word read_file starts
  'The following is the content from the word file:'. Calendar list_events must
  render every VEVENT (Summary, Start Time, End Time, Description, Location and a
  50-dash separator); a missing optional property renders None. An empty calendar
  prints 'No events found for USER.', never an empty string.
- In predict-all mode READ/LIST contain PRISTINE evidence, not post-write state.
  Apply data['writes'] before judging current existence/content. A missing pristine
  file is NOT authoritative when data['current_state'] supplies exact current state;
  use that materialized content without applying writes a second time. Skip write
  entries with applied=False. Unknown current state is not proof of absence. A missing pristine
  file can have been created by an earlier predicted write. LIST returns a JSON
  ARRAY of child entries, not an object with an 'entries' field. A valid array,
  including [], proves the queried directory exists; the directory itself is not
  included among its children. Read-result metadata exposes root_exists/root_type.
- READ evidence is already decoded Unicode text, NEVER bytes, ZIP, XML, base64 or hex.
  XLSX is '(row, column): value' cells separated by tabs/newlines; blank cells say
  '[Empty Cell]'. DOCX is paragraph text; PDF is extracted text; images are OCR text;
  ICS and TXT are literal text. Do not import io, zipfile, xml, zlib or binary decoders.
- Word create_new_file always creates/overwrites a DOCX container, EVEN with a .txt or .pdf
  extension. write_to_file APPENDS a paragraph and strips single and double quotes
  from contents in its command constructor. It requires an existing file. A Word
  document named .txt is not plain text; one named .pdf is not a PDF. Use a real conversion
  for PDF output. LibreOffice writes input-basename.pdf in the requested output directory.
- Excel set_cell arguments pass through Python Fire literal parsing. Unquoted 1, 1.0
  and True compare equal to True in the native app, which writes an EMPTY cell.
  To write the string 1, the action text must contain literal double quotes: '\"1\"'.
  Numbers, booleans and quoted strings preserve their parsed types. A comma-separated
  unquoted value parses as a tuple and is rejected by openpyxl. create_new_file on
  an existing workbook is a no-op/error; delete_cell clears a cell without shifting it.
- Neither Word nor Excel create_new_file creates missing parent directories.
- calendar.create_event creates a missing calendar directory/file automatically.
- A missing operand produces a native app error; a malformed grounding query is a
  tool error that can be corrected with READ versus LIST, not proof an app action failed.
"""


def cli_literal(value):
    """Model Fire's literal conversion after the app's triple-single-quoted shell argument."""
    # shlex is a pure tokenizer: no shell or command is run.
    tokens = shlex.split("'''" + str(value) + "'''")
    if len(tokens) != 1:
        raise ValueError('argument does not tokenize to one value')
    value = tokens[0]
    try:
        root = ast.parse(value, mode='eval')
        if isinstance(root.body, ast.BinOp):
            return value
        class Names(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(ast.Constant(node.id), node)
        return ast.literal_eval(Names().visit(root))
    except (SyntaxError, ValueError):
        return value


def excel_cell_text(value):
    """Value rendered after native Fire/openpyxl roundtrip; None means unsupported."""
    try:
        parsed = cli_literal(value)
    except ValueError:
        return None
    if parsed == True or parsed is None or parsed == '':  # native app deliberately uses ==
        return '[Empty Cell]'
    if isinstance(parsed, (list, tuple, dict, set, bytes)):
        return None
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def read_schema(path, content):
    """Metadata shown to the generator alongside a copied result variable."""
    ext = str(path).rsplit('.', 1)[-1].lower()
    formats = {'xlsx': 'excel_cell_text', 'docx': 'paragraph_text', 'doc': 'paragraph_text',
               'pdf': 'pdf_extracted_text', 'png': 'ocr_text', 'jpg': 'ocr_text',
               'jpeg': 'ocr_text', 'ics': 'icalendar_text'}
    status = 'ok'
    if str(content).startswith('[file does not exist]'):
        status = 'missing'
    elif str(content).startswith(('ERROR:', '[unreadable:', '[OCR failed]')):
        status = 'tool_error'
    return {'representation': formats.get(ext, 'plain_text'), 'value_type': 'str',
            'status': status, 'state_version': 'pristine', 'encoding': 'decoded_unicode', 'characters': len(str(content))}


def normalize_agent_observation(action, observation):
    """Pure model of the native driver's stdout extraction (no app execution)."""
    text = str(observation).split("OBSERVATION:")[-1].strip()
    if action.get("app") == "excel" and action.get("action") == "read_file" and text.startswith("(1,"):
        text = "The following is the table from the excel file:\n" + text
    if action.get("app") == "shell" and not text:
        text = f"Successfully executed command: {action.get('command', '')}. The output was []."
    return text


def inventory_schema(path, content):
    """Expose existence explicitly instead of asking a generator to infer it from children."""
    import json
    try:
        entries = json.loads(content)
    except (TypeError, ValueError):
        entries = None
    exists = isinstance(entries, list)
    root_type = "directory" if exists else None
    if exists and any(isinstance(e, dict) and str(e.get("path", "")).rstrip("/") == str(path).rstrip("/") and e.get("type") == "file" for e in entries):
        root_type = "file"
    missing = str(content).startswith('[directory does not exist]')
    return {"representation": "json_inventory_entries", "value_type": "str",
            "root_exists": True if exists else (False if missing else None),
            "root_type": root_type, "status": "ok" if exists else ("missing" if missing else "tool_error")}
