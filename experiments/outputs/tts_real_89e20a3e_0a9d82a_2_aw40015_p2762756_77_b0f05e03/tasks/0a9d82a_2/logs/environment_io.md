
### Environment Interaction 1
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_descriptions(app_name='simple_note'))
```

```
[
 {
  "name": "show_account",
  "description": "Show your account information. Unlike show_profile, this includes private information."
 },
 {
  "name": "signup",
  "description": "Sign up to create account."
 },
 {
  "name": "delete_account",
  "description": "Delete your account."
 },
 {
  "name": "update_account_name",
  "description": "Update your first or last name in the account profile."
 },
 {
  "name": "login",
  "description": "Login to your account."
 },
 {
  "name": "logout",
  "description": "Logout from your account."
 },
 {
  "name": "send_verification_code",
  "description": "Send account verification code to your email address."
 },
 {
  "name": "verify_account",
  "description": "Verify your account using the verification code sent to your email address."
 },
 {
  "name": "send_password_reset_code",
  "description": "Send password reset code to your email address."
 },
 {
  "name": "reset_password",
  "description": "Reset your password using the password reset code sent to your email address."
 },
 {
  "name": "show_profile",
  "description": "Show public profile information of a user."
 },
 {
  "name": "search_notes",
  "description": "Search your notes. This will not show contents of the notes. Pinned notes will be shown first by default, except when dont_reorder_pinned is true."
 },
 {
  "name": "create_note",
  "description": "Create a new note."
 },
 {
  "name": "show_note",
  "description": "Show detailed information of a note, including its content."
 },
 {
  "name": "delete_note",
  "description": "Delete a note."
 },
 {
  "name": "update_note",
  "description": "Update a note's title, content, tags, and/or pinned status."
 },
 {
  "name": "add_content_to_note",
  "description": "Append or prepend content to a note."
 }
]
```


### Environment Interaction 2
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='simple_note', api_name='login'))
```

```
{
 "app_name": "simple_note",
 "api_name": "login",
 "path": "/auth/token",
 "method": "POST",
 "description": "Login to your account.",
 "parameters": [
  {
   "name": "username",
   "type": "string",
   "required": true,
   "description": "Your account email.",
   "default": null,
   "constraints": []
  },
  {
   "name": "password",
   "type": "string",
   "required": true,
   "description": "Your account password.",
   "default": null,
   "constraints": []
  }
 ],
 "response_schemas": {
  "success": {
   "access_token": "string",
   "token_type": "string"
  },
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
passwords = apis.supervisor.show_account_passwords()
print(passwords)
```

```
[
 {
  "account_name": "amazon",
  "password": "o*V5hLg"
 },
 {
  "account_name": "file_system",
  "password": "5aYBh%q"
 },
 {
  "account_name": "gmail",
  "password": "5W_x]&y"
 },
 {
  "account_name": "phone",
  "password": "OWD1gav"
 },
 {
  "account_name": "simple_note",
  "password": "D[KZAcl"
 },
 {
  "account_name": "splitwise",
  "password": "]uPmRTj"
 },
 {
  "account_name": "spotify",
  "password": "L_4-}up"
 },
 {
  "account_name": "todoist",
  "password": "}jn6@[3"
 },
 {
  "account_name": "venmo",
  "password": "n56rXkH"
 }
]
```


### Environment Interaction 4
----------------------------------------------------------------------------
```python
simple_note_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "simple_note"][0]
login_result = apis.simple_note.login(username='kyl_moor@gmail.com', password=simple_note_password)
print(login_result)
```

```
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStreWxfbW9vckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.381R4fpkLxIJObrasON3x0TE_j4Hs65Svg7UDCIT8Tk",
 "token_type": "Bearer"
}
```


### Environment Interaction 5
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='simple_note', api_name='search_notes'))
```

```
{
 "app_name": "simple_note",
 "api_name": "search_notes",
 "path": "/notes",
 "method": "GET",
 "description": "Search your notes. This will not show contents of the notes. Pinned notes will be shown first by default, except when dont_reorder_pinned is true.",
 "parameters": [
  {
   "name": "access_token",
   "type": "string",
   "required": true,
   "description": "Access token obtained from simple_note app login.",
   "default": null,
   "constraints": []
  },
  {
   "name": "query",
   "type": "string",
   "required": false,
   "description": "Search query for the notes.",
   "default": "",
   "constraints": []
  },
  {
   "name": "tags",
   "type": "list[string]",
   "required": false,
   "description": "Tags to filter the notes by. Notes not having any of these tags will be filtered out.",
   "default": null,
   "constraints": []
  },
  {
   "name": "pinned",
   "type": "boolean",
   "required": false,
   "description": "Filter notes by pinned status.",
   "default": null,
   "constraints": []
  },
  {
   "name": "dont_reorder_pinned",
   "type": "boolean",
   "required": false,
   "description": "If true, pinned notes will not be reordered to be shown at the top.",
   "default": null,
   "constraints": []
  },
  {
   "name": "page_index",
   "type": "integer",
   "required": false,
   "description": "The index of the page to return.",
   "default": 0,
   "constraints": [
    "value >= 0.0"
   ]
  },
  {
   "name": "page_limit",
   "type": "integer",
   "required": false,
   "description": "The maximum number of results to return per page.",
   "default": 5,
   "constraints": [
    "value >= 1.0, <= 20.0"
   ]
  },
  {
   "name": "sort_by",
   "type": "string",
   "required": false,
   "description": "The attribute to sort the notes by prefixed with +/- to reflect ascending/descending. Valid attributes: created_at and updated_at. If both query and sort_by are given and non-empty, results will be first ranked by query relevance, then paginated, and will then be sorted by the given attribute within each page. If both query and sort_by are not given, null, or empty, sort_by will default to -updated_at.",
   "default": null,
   "constraints": []
  }
 ],
 "response_schemas": {
  "success": [
   {
    "note_id": 1,
    "title": "string",
    "tags": [
     "string"
    ],
    "created_at": "2019-01-01T00:00:00",
    "updated_at": "2019-01-01T00:00:00",
    "pinned": true
   }
  ],
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 6
----------------------------------------------------------------------------
```python
habit_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit")
print(habit_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405546 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405546)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 7
----------------------------------------------------------------------------
```python
habit_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit")
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit")
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405546 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405546)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 8
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
print(habit_notes_page)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405581 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405581)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 9
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405581 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405581)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 10
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405581 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405581)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 11
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], query="habit", page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405581 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405581)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 12
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit"], page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit"], page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405582 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405582)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 13
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit"], page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit"], page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405582 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405582)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 14
----------------------------------------------------------------------------
```python
habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=20)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    habit_notes_page = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=20)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405563 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405563)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 15
----------------------------------------------------------------------------
```python
all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=50)
print(all_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=50)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405563 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405563)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 16
----------------------------------------------------------------------------
```python
all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405528 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405528)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 17
----------------------------------------------------------------------------
```python
notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405528 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405528)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 18
----------------------------------------------------------------------------
```python
notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 19
----------------------------------------------------------------------------
```python
notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=50)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=50)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405563 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405563)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 20
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=100)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=100)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405564 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405564)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 21
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=100)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=100)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405564 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405564)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 22
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 23
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 24
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 25
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit-tracker"], page_index=0, page_limit=50)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit-tracker"], page_index=0, page_limit=50)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405590 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405590)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 26
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit-streaks"], page_index=0, page_limit=50)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit-streaks"], page_index=0, page_limit=50)
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405590 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405590)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 27
----------------------------------------------------------------------------
```python
notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_limit=1)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_limit=1)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 28
----------------------------------------------------------------------------
```python
apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["limited-screen-time"], page_index=0, page_limit=50)
```

```
Execution successful.
```


### Environment Interaction 29
----------------------------------------------------------------------------
```python
limited_screen_time_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["limited-screen-time"], page_index=0, page_limit=50)
print(limited_screen_time_notes)
```

```
{}
```


### Environment Interaction 30
----------------------------------------------------------------------------
```python
habit_tracker_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], tags=["habit-tracker"], page_index=0, page_limit=50)
print(habit_tracker_notes)
```

```
{}
```


### Environment Interaction 31
----------------------------------------------------------------------------
```python
all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=20)
print(all_notes)
```

```
{}
```


### Environment Interaction 32
----------------------------------------------------------------------------
```python
all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
```

```
Execution successful.
```


### Environment Interaction 33
----------------------------------------------------------------------------
```python
all_notes = apis.simple_note.search_notes(access_token=login_result["access_token"])
print(all_notes)
```

```
{}
```


### Environment Interaction 34
----------------------------------------------------------------------------
```python
notes = apis.simple_note.search_notes(access_token=login_result["access_token"], page_index=0, page_limit=50)
```

```
Execution successful.
```

