#!/usr/bin/env python3
"""Deliberately fail-closed example. Agents must implement geosparse_ext.entry.

Never produces synthetic training metrics or marks a model run completed.
"""
import sys
print("BLOCKED_IMPLEMENTATION: bind a real repository and implement geosparse_ext.entry",file=sys.stderr)
sys.exit(78)
