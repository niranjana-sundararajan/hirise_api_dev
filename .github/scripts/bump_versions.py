#!/usr/bin/env python3
import subprocess
import json


def get_previous_version() -> str:
    version = (
        subprocess.run(
            [
                "gh", 
                "--json", 
                "tagName", 
                "release", 
                "view"
            ],

            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf8")
        .strip()
    )

    return json.loads(version)["tagName"]


def bump_previous_version_number(version: str) -> str:
    major_version, minor_version, patch_version = version.split(".")

    return f"{major_version}.{minor_version}.{int(patch_version) + 1}"


def generate_patched_version_github_release():
    try:
        lastest_version = get_previous_version()

    except subprocess.CalledProcessError as err:
        if err.stderr.decode("utf8").startswith("HTTP 404:"):
            new_release_number = "0.0.0"
        else:
            raise Exception("Unable to generate new version number")
    else:
        new_release_number = bump_previous_version_number(lastest_version)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", new_release_number],
        check=True,
    )


if __name__ == "__main__":
    generate_patched_version_github_release()