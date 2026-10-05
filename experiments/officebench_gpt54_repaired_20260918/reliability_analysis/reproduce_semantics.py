"""Audit native OfficeBench functions on temporary files; never modify benchmark data."""
import ast
import json
import os
from pathlib import Path
import tempfile
import openpyxl
from docx import Document
from fire.parser import DefaultParseValue
from tts.env.officebench_wm_predict import symbolic_file_state

ROOT = Path('/tmp/gpt54-officebench-repaired-20260918/officebench_data/apps')

def functions(path, globals_):
    tree = ast.parse(path.read_text())
    # Native function bodies unchanged; omit unused imports (e.g. pandas) and CLI entrypoints.
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(tree, str(path), 'exec'), globals_)
    return globals_

with tempfile.TemporaryDirectory(prefix='officebench-semantics-') as tmp:
    book = str(Path(tmp) / 'shopping_list.xlsx')
    wb = openpyxl.Workbook(); wb.active.cell(2, 2, 2); wb.save(book)
    native = functions(ROOT/'excel_app/excel_set_cell.py', {'os':os,'openpyxl':openpyxl})
    parsed = DefaultParseValue('1')
    native['excel_set_cell'](book, parsed, 2, 2)
    actual = openpyxl.load_workbook(book).active.cell(2,2).value
    predicted = symbolic_file_state('data/shopping_list.xlsx', '(2, 2): 2', [(
        {'app':'excel','action':'set_cell','row_idx':2,'column_idx':2,'text':'1'}, 'ok')])
    result = {'excel_text_1': {'cli_parsed_type':type(parsed).__name__,
              'cli_parsed_value':parsed,'native_cell_value':actual,'predicted_state':predicted}}

    path = str(Path(tmp)/'lowest.txt')
    create = functions(ROOT/'word_app/word_create_new_file.py', {'Document':Document})
    write = functions(ROOT/'word_app/word_write_to_file.py', {'os':os,'Document':Document})
    assert create['word_create_new_file'](path)
    assert write['word_write_to_file'](path,'16.91')
    predicted = symbolic_file_state('data/lowest.txt','[file does not exist]',[
        ({'app':'word','action':'create_new_file','file_path':path},'ok'),
        ({'app':'word','action':'write_to_file','file_path':path,'contents':'16.91'},'ok')])
    result['word_to_txt']={'native_magic':Path(path).read_bytes()[:4].hex(),
        'native_plain_text_contains_expected': '16.91' in Path(path).read_text(errors='replace'),
        'native_docx_paragraphs':[p.text for p in Document(path).paragraphs],
        'predicted_state':predicted}
    print(json.dumps(result,indent=2))
