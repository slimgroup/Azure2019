#!/bin/bash

# 1. Upload batch_account_key to keyvault and note the account_key_keyvault_secret_id
# 2. In credentials.yaml, replace account_key with account_key_keyvault_secret_id
# 3. Use local credentials.yaml when calling shipyard and pass aad credentials
./shipyard pool list \
--keyvault-uri <value> \
--aad-directory-id <value> \
--aad-application-id <value> \
--aad-auth-key <value>
