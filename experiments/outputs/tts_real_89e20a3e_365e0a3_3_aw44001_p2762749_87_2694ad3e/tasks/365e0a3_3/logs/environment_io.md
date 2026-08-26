
### Environment Interaction 1
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_descriptions(app_name='amazon'))
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
  "name": "show_product",
  "description": "Show product information based on its ID."
 },
 {
  "name": "search_sellers",
  "description": "Search for sellers with a query."
 },
 {
  "name": "show_seller",
  "description": "Show a detailed information about the seller."
 },
 {
  "name": "search_product_types",
  "description": "Search product types present in the database."
 },
 {
  "name": "show_product_feature_choices",
  "description": "Show the choices of colors, relative sizes and sellers aggregated over all products of the given product type. Because it's an aggregation, the choices may not be available for all products. If product type is not passed, it will return the choices for all products in the database."
 },
 {
  "name": "search_products",
  "description": "Search for products with a query and various filtering criteria."
 },
 {
  "name": "show_cart",
  "description": "show your cart."
 },
 {
  "name": "clear_cart",
  "description": "Clear your cart."
 },
 {
  "name": "add_product_to_cart",
  "description": "Add product by id and quantities to your cart."
 },
 {
  "name": "delete_product_from_cart",
  "description": "Remove a product from your cart."
 },
 {
  "name": "update_product_quantity_in_cart",
  "description": "Update product quantity in the user cart."
 },
 {
  "name": "apply_promo_code_to_cart",
  "description": "Apply a promo code to your cart."
 },
 {
  "name": "remove_promo_code_from_cart",
  "description": "Remove a promo code from your cart."
 },
 {
  "name": "show_wish_list",
  "description": "Get list of products in your wishlist."
 },
 {
  "name": "clear_wish_list",
  "description": "Clear wish list."
 },
 {
  "name": "add_product_to_wish_list",
  "description": "Add product by id and quantities to your wish list."
 },
 {
  "name": "delete_product_from_wish_list",
  "description": "Remove product from the user wish list."
 },
 {
  "name": "update_product_quantity_in_wish_list",
  "description": "Update product quantity in the user wish_list."
 },
 {
  "name": "move_product_from_cart_to_wish_list",
  "description": "Move product from the cart to the wish list."
 },
 {
  "name": "move_product_from_wish_list_to_cart",
  "description": "Move product from the wish list to the cart."
 },
 {
  "name": "add_gift_wrapping_to_product",
  "description": "Add gift wrapping to a product in your cart. If the product is already set to be gift wrapped, its quantity will be updated."
 },
 {
  "name": "remove_gift_wrapping_from_product",
  "description": "Remove gift wrapping from a product in your cart."
 },
 {
  "name": "show_orders",
  "description": "Show or search your past orders"
 },
 {
  "name": "place_order",
  "description": "Place an order for all the items in your cart."
 },
 {
  "name": "show_order",
  "description": "Get details of a past order."
 },
 {
  "name": "download_order_receipt",
  "description": "Download the receipt of a past order."
 },
 {
  "name": "show_payment_cards",
  "description": "Get a list of your payment_cards."
 },
 {
  "name": "add_payment_card",
  "description": "Add a new payment card."
 },
 {
  "name": "show_payment_card",
  "description": "Get details of a payment card."
 },
 {
  "name": "delete_payment_card",
  "description": "Delete payment card information."
 },
 {
  "name": "update_payment_card",
  "description": "Update payment card information."
 },
 {
  "name": "show_addresses",
  "description": "Get a list of your addresses."
 },
 {
  "name": "add_address",
  "description": "Add a new address."
 },
 {
  "name": "delete_address",
  "description": "Delete address information."
 },
 {
  "name": "update_address",
  "description": "Update address information."
 },
 {
  "name": "show_product_reviews",
  "description": "Get a list of product reviews."
 },
 {
  "name": "write_product_review",
  "description": "Write a product review."
 },
 {
  "name": "delete_product_review",
  "description": "Delete a product review."
 },
 {
  "name": "update_product_review",
  "description": "Update a product review."
 },
 {
  "name": "show_product_questions",
  "description": "Get a list of product questions."
 },
 {
  "name": "write_product_question",
  "description": "Post a question about a product."
 },
 {
  "name": "show_product_question_answers",
  "description": "Get a list of answers to a product question."
 },
 {
  "name": "write_product_question_answer",
  "description": "Write a answer to a product question."
 },
 {
  "name": "delete_product_question",
  "description": "Delete a product question."
 },
 {
  "name": "update_product_question",
  "description": "Update a product question."
 },
 {
  "name": "delete_product_question_answer",
  "description": "Delete a answer to a product question."
 },
 {
  "name": "update_product_question_answer",
  "description": "Update answer to a product question."
 },
 {
  "name": "show_returns",
  "description": "Get a list of your product returns."
 },
 {
  "name": "initiate_return",
  "description": "Initiate a product return."
 },
 {
  "name": "show_return",
  "description": "Show product return status."
 },
 {
  "name": "show_return_deliverers",
  "description": "Get a list of product return deliverers."
 },
 {
  "name": "show_prime_plans",
  "description": "Show information about prime plans available. Delivery fee is waived for prime members."
 },
 {
  "name": "show_prime_subscriptions",
  "description": "Show your prime subscription history."
 },
 {
  "name": "subscribe_prime",
  "description": "Subscribe to prime membership."
 },
 {
  "name": "download_prime_subscription_receipt",
  "description": "Download the receipt for a prime subscription."
 }
]
```


### Environment Interaction 2
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='amazon', api_name='show_prime_subscriptions'))
```

```
{
 "app_name": "amazon",
 "api_name": "show_prime_subscriptions",
 "path": "/prime/subscriptions",
 "method": "GET",
 "description": "Show your prime subscription history.",
 "parameters": [
  {
   "name": "access_token",
   "type": "string",
   "required": true,
   "description": "Access token obtained from amazon app login.",
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
  }
 ],
 "response_schemas": {
  "success": [
   {
    "prime_subscription_id": 1,
    "start_date": "2019-01-01T00:00:00",
    "end_date": "2019-01-01T00:00:00",
    "payment_card_digits": "string",
    "paid_amount": 0.0
   }
  ],
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
passwords=apis.supervisor.show_account_passwords()
print(passwords)
```

```
[
 {
  "account_name": "amazon",
  "password": "@#@wvsw"
 },
 {
  "account_name": "file_system",
  "password": "]NVpauc"
 },
 {
  "account_name": "gmail",
  "password": "W%wsM+p"
 },
 {
  "account_name": "phone",
  "password": "vK6UAEL"
 },
 {
  "account_name": "simple_note",
  "password": "85${X=c"
 },
 {
  "account_name": "splitwise",
  "password": ")^_ixpD"
 },
 {
  "account_name": "spotify",
  "password": "@BfHM4T"
 },
 {
  "account_name": "todoist",
  "password": "eRlD2Ci"
 },
 {
  "account_name": "venmo",
  "password": "+){p}(9"
 }
]
```


### Environment Interaction 4
----------------------------------------------------------------------------
```python
amazon_password = [account["account_name"] == "amazon" for account in passwords][0]["password"]
login_result = apis.amazon.login(username='kevin.blac@gmail.com', password=amazon_password)
print(login_result)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    amazon_password = [account["account_name"] == "amazon" for account in passwords][0]["password"]
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
TypeError: 'bool' object is not subscriptable
```


### Environment Interaction 5
----------------------------------------------------------------------------
```python
amazon_password = [account["password"] for account in passwords if account["account_name"] == "amazon"][0]
print(amazon_password)
```

```
@#@wvsw
```


### Environment Interaction 6
----------------------------------------------------------------------------
```python
login_result = apis.amazon.login(username='kevin.blac@gmail.com', password=amazon_password)
print(login_result)
```

```
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ra2V2aW4uYmxhY0BnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.X4Gl4PTEwAPwHjzbef1pRJ-I8IJ_uwkjPZRkrhcme7A",
 "token_type": "Bearer"
}
```


### Environment Interaction 7
----------------------------------------------------------------------------
```python
access_token = login_result["access_token"]
page_index = 0
total_paid = 0.0
subscriptions = []
while page_index < 10:
  response = apis.amazon.show_prime_subscriptions(access_token=access_token, page_index=page_index, page_limit=20)
  if response:
    subscriptions.extend(response)
    total_paid += sum(sub["paid_amount"] for sub in response)
    if len(response) < 20:
      break
    page_index += 1
  else:
    break
print(total_paid)
```

```
Execution failed. Traceback:
  File "<python-input>", line 6, in <module>
    response = apis.amazon.show_prime_subscriptions(access_token=access_token, page_index=page_index, page_limit=20)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405566 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405566)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 8
----------------------------------------------------------------------------
```python
access_token = login_result["access_token"]
response = apis.amazon.show_prime_subscriptions(access_token=access_token, page_index=0, page_limit=20)
print(response)
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    response = apis.amazon.show_prime_subscriptions(access_token=access_token, page_index=0, page_limit=20)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405566 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405566)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 9
----------------------------------------------------------------------------
```python
response = apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'], page_index=0)
print(len(response))
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    response = apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'], page_index=0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405548 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405548)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 10
----------------------------------------------------------------------------
```python
result = apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'])
total_paid = sum(sub['paid_amount'] for sub in result)
print(total_paid)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    result = apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405531 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405531)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 11
----------------------------------------------------------------------------
```python
apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'])
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    apis.amazon.show_prime_subscriptions(access_token=login_result['access_token'])
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405531 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405531)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```

