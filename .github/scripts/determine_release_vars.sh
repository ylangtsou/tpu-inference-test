#!/bin/bash
set -eu pipefail

# --- SCHEDULE TRIGGER ---
if [[ "$GH_EVENT_NAME"  == "schedule" ]]; then
    echo "Trigger: Schedule - Generating nightly build"
    RELEASE_TYPE="nightly"

    # --- Get Base Version from Tag ---
    echo "Fetching latest tags..."
    git fetch --tags --force
    echo "Finding the latest stable version tag (vX.Y.Z)..."
    LATEST_STABLE_TAG=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)
    if [[ -z "$LATEST_STABLE_TAG" ]]; then
    echo "Warning: No stable tag found. Using default 0.0.0 as base."
    BASE_VERSION="0.1.0"
    else
    BASE_VERSION=${LATEST_STABLE_TAG#v}
    fi
    echo "Using BASE_VERSION=${BASE_VERSION}"


    # --- Generate Nightly Version ---
    DATETIME_STR=$(date -u +%Y%m%d%H%M)
    VERSION="${BASE_VERSION}.dev${DATETIME_STR}"


# --- MANUAL DISPATCH TRIGGER ---
elif [[ "$GH_EVENT_NAME" == "workflow_dispatch" ]]; then
    echo "Trigger: Manual Dispatch"
    RELEASE_TYPE="$INPUT_RELEASE_TYPE"

    if [[ "$RELEASE_TYPE" == "nightly" ]]; then
    if [[ -z "$INPUT_VERSION" ]]; then
        echo "Error: Manual stable release requires a version input."
        exit 1
    fi
    BASE_VERSION="$INPUT_VERSION"

    DATETIME_STR=$(date -u +%Y%m%d%H%M)
    VERSION="${BASE_VERSION}.dev${DATETIME_STR}"
    else
    RELEASE_TYPE="stable"

    if [[ -z "$INPUT_VERSION" ]]; then
        echo "Error: Manual stable release requires a version input."
        exit 1
    fi
    VERSION="$INPUT_VERSION"
    fi

# --- PUSH TAG TRIGGER ---
elif [[ "$GH_EVENT_NAME" == "push" && "$GH_REF" == refs/tags/* ]]; then
    echo "Trigger: Push Tag"
    RELEASE_TYPE="stable"
    TAG_NAME="$GH_REF_NAME"
    VERSION=${TAG_NAME#v}

# --- PUSH BRANCH TRIGGER ---
elif [[ "$GH_EVENT_NAME" == "push" && "$GH_REF" == refs/heads/* ]]; then
    echo "Trigger: Push Branch (${GH_REF_NAME}) - Defaulting to nightly 0.1.0 base"
    RELEASE_TYPE="nightly"
    BASE_VERSION="0.1.0"
    echo "Using default BASE_VERSION=${BASE_VERSION}"
    DATETIME_STR=$(date -u +%Y%m%d%H%M)
    VERSION="${BASE_VERSION}.dev${DATETIME_STR}"

# --- ERROR HANDLING ---
else
    echo "Error: Unknown or unsupported trigger."
    exit 1
fi

# --- output ---
echo "Final determined values: RELEASE_TYPE=${RELEASE_TYPE:-nightly}, VERSION=${VERSION}"
echo "RELEASE_TYPE=${RELEASE_TYPE:-nightly}" >> "$GITHUB_OUTPUT"
echo "VERSION=${VERSION}" >> "$GITHUB_OUTPUT"
