from pathlib import Path
from typing import Union

from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.simplerunner.NegoRunner import StdOutReporter
from uri.uri import URI


def get_domains(domains_dir: Union[str, Path]) -> tuple[tuple[URI, URI]]:
    dir = Path(domains_dir)

    A_profiles = (URI(f"file:{x.as_posix()}") for x in sorted(dir.glob("domain*/profileA.json")))
    B_profiles = (URI(f"file:{x.as_posix()}") for x in sorted(dir.glob("domain*/profileB.json")))
    domains = tuple(zip(A_profiles, B_profiles))

    return domains


def get_utility_function(profile_uri: URI) -> LinearAdditiveUtilitySpace:
    """Load geniusweb utility class based on the file URI

    Args:
        profile_uri (URI): URI of preference profile file

    Returns:
        LinearAdditiveUtilitySpace: Utility class based on preference specification
    """
    profile_connection = ProfileConnectionFactory.create(profile_uri, StdOutReporter())
    profile = profile_connection.getProfile()
    profile_connection.close()
    assert isinstance(profile, LinearAdditiveUtilitySpace)

    return profile
