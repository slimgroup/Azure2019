#!/bin/bash

# 1. Prepare credentials.yaml with all ids and passwords
# 2. Upload credentials.yaml using the upload_credentials.sh shell script
# Fetch full credentials.yaml from keyvault (remove credentials.yaml from current directory)
./shipyard pool list \
--keyvault-uri <value> \
--keyvault-credentials-secret-id <value> \
--aad-directory-id <value> \
--aad-application-id <value> \
--aad-auth-key <value>
