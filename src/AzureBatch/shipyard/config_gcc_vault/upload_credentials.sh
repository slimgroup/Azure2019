#!/bin/bash

# Store credentials.yaml in keyvault as mycreds
./shipyard keyvault add mycredentials \
--keyvault-uri <value> \
--aad-directory-id <value> \
--aad-application-id <value> \
--aad-auth-key <value> \
--credentials credentials.yaml