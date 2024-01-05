#!/usr/bin/env bash

rsync \
  --archive \
  --verbose \
  --exclude '.git' \
  --exclude '.ruff_cache' \
  o2:dbp/data/smp/20231229_compare_field_types/ .
