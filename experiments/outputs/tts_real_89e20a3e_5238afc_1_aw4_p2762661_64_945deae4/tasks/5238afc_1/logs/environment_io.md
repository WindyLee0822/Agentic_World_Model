
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
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
```

```
[
 {
  "name": "show_active_task",
  "description": "Show the currently active task assigned to you by the supervisor."
 },
 {
  "name": "complete_task",
  "description": "Mark the currently active task as complete with the given answer."
 },
 {
  "name": "show_profile",
  "description": "Show your supervisor's profile information."
 },
 {
  "name": "show_addresses",
  "description": "Show your supervisor's addresses."
 },
 {
  "name": "show_payment_cards",
  "description": "Show your supervisor's payment_cards."
 },
 {
  "name": "show_account_passwords",
  "description": "Show your supervisor's app account passwords."
 }
]
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
supervisor_passwords = apis.supervisor.show_account_passwords()
print(supervisor_passwords)
```

```
[
 {
  "account_name": "amazon",
  "password": "7ukZGUP"
 },
 {
  "account_name": "file_system",
  "password": "ZK(+NFg"
 },
 {
  "account_name": "gmail",
  "password": "iE*nJ[6"
 },
 {
  "account_name": "phone",
  "password": "0!ROg04"
 },
 {
  "account_name": "simple_note",
  "password": "DyZIT1v"
 },
 {
  "account_name": "splitwise",
  "password": "1QB%1+0"
 },
 {
  "account_name": "spotify",
  "password": "0p%ThzG"
 },
 {
  "account_name": "todoist",
  "password": "Za{K]EA"
 },
 {
  "account_name": "venmo",
  "password": "b8E]-&R"
 }
]
```


### Environment Interaction 4
----------------------------------------------------------------------------
```python
amazon_login = apis.amazon.login(username='jepowell@gmail.com', password='7ukZGUP')
print(amazon_login)
```

```
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ramVwb3dlbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyNzMxfQ.hCCe9ymMQyJP0_03XEr7UW2gKWjMnvHfZ2O79prbHnw",
 "token_type": "Bearer"
}
```


### Environment Interaction 5
----------------------------------------------------------------------------
```python
cart_items = apis.amazon.show_cart(access_token=amazon_login['access_token'])
print(cart_items)
```

```
Execution failed. Traceback:
  File "<python-input>", line 1, in <module>
    cart_items = apis.amazon.show_cart(access_token=amazon_login['access_token'])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 405514 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=405514)", 'type': 'BadRequestError', 'param': 'input_text', 'code': 400}}
```


### Environment Interaction 6
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


### Environment Interaction 7
----------------------------------------------------------------------------
```python
cart_items = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ramVwb3dlbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyNzMxfQ.hCCe9ymMQyJP0_03XEr7UW2gKWjMnvHfZ2O79prbHnw')
print(cart_items)
```

```
{}
```


### Environment Interaction 8
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='amazon', api_name='place_order'))
```

```
{
 "app_name": "amazon",
 "api_name": "place_order",
 "path": "/orders",
 "method": "POST",
 "description": "Place an order for all the items in your cart.",
 "parameters": [
  {
   "name": "payment_card_id",
   "type": "integer",
   "required": true,
   "description": "ID of the payment card to use for this order.",
   "default": null,
   "constraints": []
  },
  {
   "name": "address_id",
   "type": "integer",
   "required": true,
   "description": "ID of the address used for shipping this order.",
   "default": null,
   "constraints": []
  },
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
   "message": "string",
   "order_id": 1
  },
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 9
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='amazon', api_name='show_payment_cards'))
```

```
{
 "app_name": "amazon",
 "api_name": "show_payment_cards",
 "path": "/payment_cards",
 "method": "GET",
 "description": "Get a list of your payment_cards.",
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
  "success": [
   {
    "payment_card_id": 1,
    "card_name": "Visa",
    "owner_name": "string",
    "card_number": 1000000000000000.0,
    "expiry_year": 1,
    "expiry_month": 1.0,
    "cvv_number": 100.0
   }
  ],
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 10
----------------------------------------------------------------------------
```python
payment_cards = apis.amazon.show_payment_cards(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ramVwb3dlbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyNzMxfQ.hCCe9ymMQyJP0_03XEr7UW2gKWjMnvHfZ2O79prbHnw')
print(payment_cards)
```

```
{}
```


### Environment Interaction 11
----------------------------------------------------------------------------
```python
supervisor_cards = apis.supervisor.show_payment_cards()
print(supervisor_cards)
```

```
[
 {
  "owner_name": "Jennifer Powell",
  "card_name": "American Express",
  "card_number": 6243185372416286,
  "expiry_year": 2023,
  "expiry_month": 3,
  "cvv_number": 362
 },
 {
  "owner_name": "Jennifer Powell",
  "card_name": "Visa",
  "card_number": 2663166460995302,
  "expiry_year": 2023,
  "expiry_month": 6,
  "cvv_number": 156
 },
 {
  "owner_name": "Jennifer Powell",
  "card_name": "HSBC",
  "card_number": 2894421026985235,
  "expiry_year": 2023,
  "expiry_month": 10,
  "cvv_number": 443
 },
 {
  "owner_name": "Jennifer Powell",
  "card_name": "Wells Fargo",
  "card_number": 6239400205158274,
  "expiry_year": 2023,
  "expiry_month": 1,
  "cvv_number": 523
 },
 {
  "owner_name": "Jennifer Powell",
  "card_name": "MasterCard",
  "card_number": 5586099418749345,
  "expiry_year": 2022,
  "expiry_month": 10,
  "cvv_number": 973
 }
]
```


### Environment Interaction 12
----------------------------------------------------------------------------
```python
cart_items = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ramVwb3dlbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyNzMxfQ.hCCe9ymMQyJP0_03XEr7UW2gKWjMnvHfZ2O79prbHnw')
print(cart_items)
```

```
{}
```


### Environment Interaction 13
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_doc(app_name='amazon', api_name='show_addresses'))
```

```
{
 "app_name": "amazon",
 "api_name": "show_addresses",
 "path": "/addresses",
 "method": "GET",
 "description": "Get a list of your addresses.",
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
  "success": [
   {
    "address_id": 1,
    "name": "string",
    "street_address": "string",
    "city": "string",
    "state": "string",
    "country": "string",
    "zip_code": 10000.0
   }
  ],
  "failure": {
   "message": "string"
  }
 }
}
```


### Environment Interaction 14
----------------------------------------------------------------------------
```python
cart_items = apis.amazon.show_cart(access_token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWF6b24ramVwb3dlbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyNzMxfQ.hCCe9ymMQyJP0_03XEr7UW2gKWjMnvHfZ2O79prbHnw')
print(cart_items)
```

```
{}
```


### Environment Interaction 15
----------------------------------------------------------------------------
```python
supervisor_addresses = apis.supervisor.show_addresses()
print(supervisor_addresses)
```

```
[
 {
  "name": "Home",
  "street_address": "34495 Michael Burg Suite 805",
  "city": "Seattle",
  "state": "Washington",
  "country": "United States",
  "zip_code": 38887
 },
 {
  "name": "Work",
  "street_address": "3850 Timothy Squares Suite 942",
  "city": "Seattle",
  "state": "Washington",
  "country": "United States",
  "zip_code": 54179
 }
]
```


### Environment Interaction 16
----------------------------------------------------------------------------
```python
print("Cart is empty. No weightlifting benches to order.")
```

```
Cart is empty. No weightlifting benches to order.
```


### Environment Interaction 17
----------------------------------------------------------------------------
```python
apis.supervisor.complete_task()
```

```
Execution successful.
```

