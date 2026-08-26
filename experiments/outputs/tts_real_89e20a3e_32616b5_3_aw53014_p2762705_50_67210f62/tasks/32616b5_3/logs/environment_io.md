
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
print(apis.api_docs.show_api_descriptions(app_name='splitwise'))
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
  "name": "search_users",
  "description": "Search Splitwise users by name or email address."
 },
 {
  "name": "show_activity",
  "description": "Show a history of your expenses and payments combined."
 },
 {
  "name": "show_groups",
  "description": "Get a list of groups you are a member of."
 },
 {
  "name": "create_group",
  "description": "Create a new group of friends or family to share expenses with."
 },
 {
  "name": "show_group",
  "description": "Show group details based on its ID."
 },
 {
  "name": "delete_group",
  "description": "Delete a group you are a member of. Any member can undelete it later."
 },
 {
  "name": "update_group",
  "description": "Update group information."
 },
 {
  "name": "undelete_group",
  "description": "Undelete a group you are a member of."
 },
 {
  "name": "add_member_to_group",
  "description": "Add a member to a group."
 },
 {
  "name": "remove_member_from_group",
  "description": "Remove a member from a group."
 },
 {
  "name": "exit_group",
  "description": "Exit from a group you are a part of."
 },
 {
  "name": "regenerate_invitation_code",
  "description": "Regenerate the invitation code for a group. Anyone with this code can join the group. The old code will be invalidated."
 },
 {
  "name": "accept_group_invitation",
  "description": "Accept a group invitation shared with you by one of its members."
 },
 {
  "name": "record_expense",
  "description": "Record a new expense to share with others."
 },
 {
  "name": "download_expense_receipt_file",
  "description": "Download a receipt file attachment for an expense."
 },
 {
  "name": "attach_expense_receipt_file",
  "description": "Attach a receipt file to an expense."
 },
 {
  "name": "delete_expense_receipt_file",
  "description": "Delete the receipt file from an expense."
 },
 {
  "name": "show_expense",
  "description": "Show detailed expense information based on its ID."
 },
 {
  "name": "delete_expense",
  "description": "Mark the expense you are involved in as deleted. Anyone involved in the expense or a member of its group can undelete it later."
 },
 {
  "name": "update_expense",
  "description": "Update expense information."
 },
 {
  "name": "show_group_expenses",
  "description": "Show or search your expenses from a given group based on various criteria."
 },
 {
  "name": "show_no_group_expenses",
  "description": "Show or search your expenses that are not part of any group based on various criteria."
 },
 {
  "name": "undelete_expense",
  "description": "Restore a previously deleted expense you are involved in."
 },
 {
  "name": "show_expense_comments",
  "description": "Get a list of expense comments."
 },
 {
  "name": "post_expense_comment",
  "description": "Post a a comment on an expense."
 },
 {
  "name": "show_expense_comment",
  "description": "Get information about a comment posted on an expense."
 },
 {
  "name": "delete_expense_comment",
  "description": "Delete a comment you posted on an expense."
 },
 {
  "name": "update_expense_comment",
  "description": "Update a comment you posted on an expense."
 },
 {
  "name": "record_payment",
  "description": "Record a new payment for an expense. This only records payment on splitwise and does not move real money."
 },
 {
  "name": "download_payment_receipt_file",
  "description": "Download receipt file attachment for a payment."
 },
 {
  "name": "attach_payment_receipt_file",
  "description": "Attach a receipt file to a payment."
 },
 {
  "name": "delete_payment_receipt_file",
  "description": "Delete the receipt file from a payment."
 },
 {
  "name": "show_payment",
  "description": "Show payment details based on its ID."
 },
 {
  "name": "delete_payment",
  "description": "Mark the payment you are involved in as deleted. Anyone involved in the payment or a member of its group can undelete it later."
 },
 {
  "name": "update_payment",
  "description": "Update payment information."
 },
 {
  "name": "show_group_payments",
  "description": "Show all the payments that are part of the given group."
 },
 {
  "name": "show_no_group_payments",
  "description": "Show payments you are involved in that are not part of any group."
 },
 {
  "name": "undelete_payment",
  "description": "Restore a previously deleted payment you are involved in."
 },
 {
  "name": "show_person_balance",
  "description": "Show the amounts you and a given person owe to each other, broken down by group. The non-grouped expenses/payments will be shown as a group with ID None."
 },
 {
  "name": "show_people_balance",
  "description": "Show the aggregate amounts you owe to each person and they owe to you."
 },
 {
  "name": "show_group_balance",
  "description": "For the given group, show the detailed breakdown of amounts different members owe to each other. If group_id is not passed, show the amounts you owe to others and vice-versa for non-grouped expenses/payments."
 },
 {
  "name": "show_groups_balance",
  "description": "Show the aggregate amounts you owe to others or others owe you for each group you are member of. The non-grouped expenses/payments will be shown as a group with group_id of None."
 },
 {
  "name": "settle_up",
  "description": "Settle up outstanding balance with a user in a group. This only records payment on splitwise and does not move real money."
 },
 {
  "name": "show_payment_comments",
  "description": "Get a list of payment comments."
 },
 {
  "name": "post_payment_comment",
  "description": "Create a new payment comment."
 },
 {
  "name": "show_payment_comment",
  "description": "Get information about a comment posted on a payment."
 },
 {
  "name": "delete_payment_comment",
  "description": "Delete a comment you posted on a payment."
 },
 {
  "name": "update_payment_comment",
  "description": "Update a payment comment."
 },
 {
  "name": "show_notifications",
  "description": "Get a list of your notifications."
 },
 {
  "name": "delete_notifications",
  "description": "Delete all of your notifications."
 },
 {
  "name": "mark_notifications",
  "description": "Mark all notifications as read or unread."
 },
 {
  "name": "show_notifications_count",
  "description": "Get the count of your notifications."
 },
 {
  "name": "delete_notification",
  "description": "Delete a notification."
 },
 {
  "name": "mark_notification",
  "description": "Mark a notification as read or unread."
 }
]
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
  "password": "VxoU[nz"
 },
 {
  "account_name": "file_system",
  "password": "]qH1!p3"
 },
 {
  "account_name": "gmail",
  "password": "%@g}L&b"
 },
 {
  "account_name": "phone",
  "password": "ct_dP=D"
 },
 {
  "account_name": "simple_note",
  "password": "ynNV=(s"
 },
 {
  "account_name": "splitwise",
  "password": "*!Zvmyo"
 },
 {
  "account_name": "spotify",
  "password": "(dgokRb"
 },
 {
  "account_name": "todoist",
  "password": "VK^N(lj"
 },
 {
  "account_name": "venmo",
  "password": "{Ew}9!q"
 }
]
```


### Environment Interaction 4
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


### Environment Interaction 5
----------------------------------------------------------------------------
```python
simple_note_password = [p["password"] for p in passwords if p["account_name"] == "simple_note"][0]
simple_note_login = apis.simple_note.login(username='les_ball@gmail.com', password=simple_note_password)
print(simple_note_login)
```

```
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStsZXNfYmFsbEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.s0oYmsr1UQn7jGLkgspCtFVj-XPSIEmbdbOqXJ44N4g",
 "token_type": "Bearer"
}
```


### Environment Interaction 6
----------------------------------------------------------------------------
```python
search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
print(search_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405528 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405528)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 7
----------------------------------------------------------------------------
```python
search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
print(search_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405528 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405528)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 8
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


### Environment Interaction 9
----------------------------------------------------------------------------
```python
search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"], query="trip")
print(search_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"], query="trip")
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405545 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405545)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 10
----------------------------------------------------------------------------
```python
search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"], page_index=0, page_limit=20)
print(search_notes)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"], page_index=0, page_limit=20)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405563 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405563)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 11
----------------------------------------------------------------------------
```python
search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    search_notes = apis.simple_note.search_notes(access_token=simple_note_login["access_token"])
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405528 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405528)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 12
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='simple_note', api_name='show_note'))
```

```
{
 "app_name": "simple_note",
 "api_name": "show_note",
 "path": "/notes/{note_id}",
 "method": "GET",
 "description": "Show detailed information of a note, including its content.",
 "parameters": [
  {
   "name": "note_id",
   "type": "integer",
   "required": true,
   "description": "ID of the note to be shown.",
   "default": null,
   "constraints": []
  },
  {
   "name": "access_token",
   "type": "string",
   "required": true,
   "description": "Access token obtained from simple_note app login.",
   "default": null,
   "constraints": []
  }
 ],
 "response_schemas": {
  "success": {
   "note_id": 1,
   "title": "string",
   "content": "string",
   "tags": [
    "string"
   ],
   "created_at": "2019-01-01T00:00:00",
   "updated_at": "2019-01-01T00:00:00",
   "pinned": true
  },
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 13
----------------------------------------------------------------------------
```python
note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
print(note1)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405539 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405539)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 14
----------------------------------------------------------------------------
```python
note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405539 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405539)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 15
----------------------------------------------------------------------------
```python
note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    note1 = apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405539 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405539)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 16
----------------------------------------------------------------------------
```python
apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.show_note(note_id=1, access_token=simple_note_login["access_token"])
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405539 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405539)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 17
----------------------------------------------------------------------------
```python
apis.simple_note.show_note(note_id=1, access_token=simple_note_login['access_token'])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.simple_note.show_note(note_id=1, access_token=simple_note_login['access_token'])
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405539 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405539)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```

