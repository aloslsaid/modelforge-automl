# Security Policy

## Reporting a vulnerability

Please do not publish secrets, private datasets, or exploitable vulnerability details in a public issue. Contact the repository maintainer privately through GitHub's security reporting feature when available.

## Data and credentials

- Keep provider keys in an untracked `.env` file; never commit real credentials.
- Do not send confidential or regulated datasets to an external model provider without authorization.
- Treat AI-generated cleaning plans as untrusted input. The application validates requested columns and protects the selected target column.
- Run uploads only in a trusted environment and remove generated files when they are no longer needed.
