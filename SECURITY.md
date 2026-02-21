# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it through GitHub's private vulnerability reporting feature:

1. Go to the [Security Advisories](https://github.com/brandonmbehring-dev/ir-eval/security/advisories) page
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 5 business days
- **Resolution target**: Within 30 days for critical issues

## Security Considerations

ir-eval processes JSON/YAML files and numerical arrays for evaluation metrics.
The library:

- Does NOT store or transmit user data beyond local files
- Does NOT make network requests (the core library; adapters may)
- Does NOT execute arbitrary code from inputs
- Operates entirely in-memory with file I/O for golden sets and results

### Dependencies

Security updates to dependencies are incorporated in patch releases.
