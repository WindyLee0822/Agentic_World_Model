
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
# First, get the Amazon password from supervisor app
passwords = apis.supervisor.show_account_passwords()
amazon_password = [p["password"] for p in passwords if p["account_name"] == "amazon"][0]
print(f"Amazon password: {amazon_password}")
```

```
Amazon password: hq(Y-=i
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
amazon_login = apis.amazon.login(username='jo-harr@gmail.com', password='hq(Y-=i')
print(amazon_login)
```

```
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI3MzF9.4zPH4lVkF5ZbhunPsrqgp39apiUlchvBQajBi-wz6-0",
 "token_type": "Bearer"
}
```


### Environment Interaction 4
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='amazon', api_name='show_cart'))
```

```
{
 "app_name": "amazon",
 "api_name": "show_cart",
 "path": "/cart",
 "method": "GET",
 "description": "show your cart.",
 "parameters": [
  {
   "name": "access_token",
   "type": "string",
   "required": true,
   "description": "Access token obtained from amazon app login.",
   "default": null,
   "constraints": []
  }
 ],
 "response_schemas": {
  "success": {
   "delivery_fee": 0.0,
   "base_cost": 0.0,
   "tax": 0.0,
   "gift_wrap_fee": 0.0,
   "discount": 0.0,
   "total_cost": 0.0,
   "promo_code": "string",
   "promo_valid": true,
   "cart_items": [
    {
     "product_id": 1,
     "product_name": "string",
     "delivery_days": 1,
     "quantity": 1,
     "price": 0.0,
     "gift_wrap_quantity": 1
    }
   ]
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
amazon_access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI3MzF9.4zPH4lVkF5ZbhunPsrqgp39apiUlchvBQajBi-wz6-0"
cart = apis.amazon.show_cart(access_token=amazon_access_token)
print(cart)
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    cart = apis.amazon.show_cart(access_token=amazon_access_token)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405512 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405512)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 6
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token=amazon_access_token)
print(cart)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    cart = apis.amazon.show_cart(access_token=amazon_access_token)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405512 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405512)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 7
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI3MzF9.4zPH4lVkF5ZbhunPsrqgp39apiUlchvBQajBi-wz6-0')
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI3MzF9.4zPH4lVkF5ZbhunPsrqgp39apiUlchvBQajBi-wz6-0')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405512 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405512)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 8
----------------------------------------------------------------------------
```python
amazon_login = apis.amazon.login(username='jo-harr@gmail.com', password='hq(Y-=i')
print(amazon_login.get("access_token"))
```

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIxNTd9.iHRJfrJ7LcvIh71UecaPaZZKfGmdPwwGYDEOT9QiNVw
```


### Environment Interaction 9
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIxNTd9.iHRJfrJ7LcvIh71UecaPaZZKfGmdPwwGYDEOT9QiNVw')
print(cart.get("base_cost"))
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIxNTd9.iHRJfrJ7LcvIh71UecaPaZZKfGmdPwwGYDEOT9QiNVw')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405512 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405512)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 10
----------------------------------------------------------------------------
```python
amazon_token = apis.amazon.login(username='jo-harr@gmail.com', password='hq(Y-=i')["access_token"]
print(amazon_token)
```

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI2MDR9.BcAKjWlqNLhr6ZDhXxpWaQASthhHet-wNFMS3JmH57U
```


### Environment Interaction 11
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI2MDR9.BcAKjWlqNLhr6ZDhXxpWaQASthhHet-wNFMS3JmH57U')
print(cart)
```

```
{}
```


### Environment Interaction 12
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_descriptions(app_name='venmo'))
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
  "description": "Show public profile information of a user, including your friendship status with them."
 },
 {
  "name": "search_users",
  "description": "Search Venmo users by name or email address."
 },
 {
  "name": "search_friends",
  "description": "Search your or others' friends by name or email address."
 },
 {
  "name": "add_friend",
  "description": "Add a friend to your friend list."
 },
 {
  "name": "remove_friend",
  "description": "Remove a friend from your friend list."
 },
 {
  "name": "show_venmo_balance",
  "description": "Show your Venmo balance."
 },
 {
  "name": "add_to_venmo_balance",
  "description": "Add money to your Venmo balance."
 },
 {
  "name": "withdraw_from_venmo_balance",
  "description": "Withdraw money from your Venmo balance."
 },
 {
  "name": "show_bank_transfer_history",
  "description": "Show histroy of money transfer from Venmo to payment card and vice versa."
 },
 {
  "name": "download_bank_transfer_receipt",
  "description": "Download the receipt of money transfer from Venmo to payment card or vice versa."
 },
 {
  "name": "show_transaction",
  "description": "Show transaction details."
 },
 {
  "name": "update_transaction",
  "description": "Update transaction information."
 },
 {
  "name": "show_transactions",
  "description": "Get a list of your transactions."
 },
 {
  "name": "create_transaction",
  "description": "Send money to a user."
 },
 {
  "name": "download_transaction_receipt",
  "description": "Download the receipt of a transaction (money sent from one user to another)."
 },
 {
  "name": "like_transaction",
  "description": "Like a transaction."
 },
 {
  "name": "unlike_transaction",
  "description": "Unlike a transaction."
 },
 {
  "name": "show_transaction_comments",
  "description": "Get a list of transaction comments."
 },
 {
  "name": "create_transaction_comment",
  "description": "Create a new transaction comment."
 },
 {
  "name": "show_transaction_comment",
  "description": "Show detailed information about a transaction comment."
 },
 {
  "name": "delete_transaction_comment",
  "description": "Delete a transaction comment."
 },
 {
  "name": "update_transaction_comment",
  "description": "Update a transaction comment."
 },
 {
  "name": "like_transaction_comment",
  "description": "Like a transaction comment."
 },
 {
  "name": "unlike_transaction_comment",
  "description": "Unlike a previously liked transaction comment."
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
  "name": "show_payment_cards",
  "description": "Get a list of users payment cards."
 },
 {
  "name": "add_payment_card",
  "description": "Add a new payment card."
 },
 {
  "name": "show_received_payment_requests",
  "description": "Get a list of payment requests you have received from others."
 },
 {
  "name": "show_sent_payment_requests",
  "description": "Get a list of payment requests you have sent to others."
 },
 {
  "name": "create_payment_request",
  "description": "Send a payment request."
 },
 {
  "name": "delete_payment_request",
  "description": "Delete a payment request."
 },
 {
  "name": "update_payment_request",
  "description": "Update payment request information."
 },
 {
  "name": "approve_payment_request",
  "description": "Approve a payment request."
 },
 {
  "name": "deny_payment_request",
  "description": "Deny a payment request."
 },
 {
  "name": "remind_payment_request",
  "description": "Send a reminder to a user via notification about this payment request."
 },
 {
  "name": "show_social_feed",
  "description": "Show your social feed (transactions of your friends)."
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


### Environment Interaction 13
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_descriptions(app_name='phone'))
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
  "name": "send_password_reset_code",
  "description": "Send password reset code to your phone number."
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
  "name": "show_contact_relationships",
  "description": "Get a list of all relationships available in your contact book."
 },
 {
  "name": "search_contacts",
  "description": "Search your contact book for relatives' information."
 },
 {
  "name": "add_contact",
  "description": "Add a new contact."
 },
 {
  "name": "delete_contact",
  "description": "Delete contact information."
 },
 {
  "name": "update_contact",
  "description": "Update contact information."
 },
 {
  "name": "show_text_message_window",
  "description": "Show text messages with a contact around a given date and time."
 },
 {
  "name": "search_text_messages",
  "description": "Show or search your text messages."
 },
 {
  "name": "show_text_message",
  "description": "Show text message details."
 },
 {
  "name": "delete_text_message",
  "description": "Delete a text message."
 },
 {
  "name": "send_text_message",
  "description": "Send a text message on the given phone number."
 },
 {
  "name": "show_alarms",
  "description": "Get a list of alarms."
 },
 {
  "name": "create_alarm",
  "description": "Create a new alarm."
 },
 {
  "name": "show_alarm",
  "description": "Show alarm details."
 },
 {
  "name": "delete_alarm",
  "description": "Delete an alarm."
 },
 {
  "name": "update_alarm",
  "description": "Update an alarm's settings."
 },
 {
  "name": "show_voice_message_window",
  "description": "Show voice messages with a contact around a given date and time."
 },
 {
  "name": "search_voice_messages",
  "description": "Show or search text voice_messages between the user and a contact."
 },
 {
  "name": "show_voice_message",
  "description": "Show voice message details."
 },
 {
  "name": "delete_voice_message",
  "description": "Delete a voice message."
 },
 {
  "name": "send_voice_message",
  "description": "Send a voice message on the given phone number."
 },
 {
  "name": "get_current_date_and_time",
  "description": "Show current date and time."
 }
]
```


### Environment Interaction 14
----------------------------------------------------------------------------
```python
relationships = apis.phone.show_contact_relationships()
print(relationships)
```

```
{}
```


### Environment Interaction 15
----------------------------------------------------------------------------
```python
troy_contact = apis.phone.search_contacts(query="Troy")
print(troy_contact)
```

```
{}
```


### Environment Interaction 16
----------------------------------------------------------------------------
```python
contacts = apis.phone.search_contacts(query="Troy")
if contacts:
    for contact in contacts:
        print(f"Name: {contact.get('name')}, Phone: {contact.get('phone')}")
```

```
Execution successful.
```


### Environment Interaction 17
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI2MDR9.BcAKjWlqNLhr6ZDhXxpWaQASthhHet-wNFMS3JmH57U')
print(cart.get("base_cost"))
```

```
None
```


### Environment Interaction 18
----------------------------------------------------------------------------
```python
cart = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ram8taGFyckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTI2MDR9.BcAKjWlqNLhr6ZDhXxpWaQASthhHet-wNFMS3JmH57U')
print(cart)
```

```
{}
```


### Environment Interaction 19
----------------------------------------------------------------------------
```python
passwords = apis.supervisor.show_account_passwords()
amazon_password = [p["password"] for p in passwords if p["account_name"] == "amazon"][0]
print(f"Amazon password: {amazon_password}")
```

```
Amazon password: hq(Y-=i
```


### Environment Interaction 20
----------------------------------------------------------------------------
```python
troy_contacts = apis.phone.search_contacts(query="Troy")
if troy_contacts and isinstance(troy_contacts, list):
    for contact in troy_contacts:
        if contact.get("name") == "Troy":
            print(f"Troy email: {contact.get('email')}")
            print(f"Troy phone: {contact.get('phone')}")
            break
```

```
Execution successful.
```


### Environment Interaction 21
----------------------------------------------------------------------------
```python
venmo_passwords = apis.supervisor.show_account_passwords()
venmo_password = [p["password"] for p in venmo_passwords if p["account_name"] == "venmo"][0]
print(f"Venmo password: {venmo_password}")
```

```
Venmo password: ZEgU=ep
```


### Environment Interaction 22
----------------------------------------------------------------------------
```python
amazon_token = apis.amazon.login(username='jo-harr@gmail.com', password='hq(Y-=i')["access_token"]
cart = apis.amazon.show_cart(access_token=amazon_token)
print(cart.get("base_cost", 0))
```

```
0
```


### Environment Interaction 23
----------------------------------------------------------------------------
```python
cart_items = apis.amazon.show_cart(access_token=amazon_token).get("cart_items", [])
print(f"Number of items in cart: {len(cart_items)}")
```

```
Number of items in cart: 0
```


### Environment Interaction 24
----------------------------------------------------------------------------
```python
print(apis.amazon.show_cart(access_token=amazon_token))
```

```
{}
```


### Environment Interaction 25
----------------------------------------------------------------------------
```python
troy_contact = apis.phone.search_contacts(query="Troy")
if troy_contact and isinstance(troy_contact, dict) and troy_contact.get("name") == "Troy":
    print(f"Troy email: {troy_contact.get('email')}")
    print(f"Troy phone: {troy_contact.get('phone')}")
```

```
Execution successful.
```


### Environment Interaction 26
----------------------------------------------------------------------------
```python
# Get Troy's contact info
troy_contact = apis.phone.search_contacts(query="Troy")
if troy_contact and isinstance(troy_contact, list):
    for contact in troy_contact:
        if contact.get("name") == "Troy":
            print(f"Troy found: {contact}")
elif troy_contact and isinstance(troy_contact, dict):
    print(f"Troy found: {troy_contact}")
else:
    print("Troy not found in contacts")
```

```
Troy not found in contacts
```

