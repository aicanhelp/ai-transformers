#!/usr/bin/env bash

echo "Copy model from /app/models to $1"
cp -R /app/models/* $1