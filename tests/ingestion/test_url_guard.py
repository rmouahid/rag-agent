import socket

import pytest

from rag_agent.ingestion import url_guard
from rag_agent.ingestion.url_guard import BlockedURLError, check_url


@pytest.fixture
def resolves_to(monkeypatch):
    """Makes every host resolve to the given addresses, without DNS."""

    def set_addresses(*addresses):
        def getaddrinfo(host, port, proto=0):
            return [
                (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, proto, "", (a, port))
                for a in addresses
            ]

        monkeypatch.setattr(url_guard.socket, "getaddrinfo", getaddrinfo)

    return set_addresses


def test_public_address_is_allowed(resolves_to):
    resolves_to("93.184.215.14")

    check_url("https://example.com/page")


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",        # loopback
        "10.0.0.5",         # RFC 1918
        "172.16.3.4",
        "192.168.1.10",
        "169.254.169.254",  # link-local, cloud metadata
        "100.64.0.1",       # shared address space (CGNAT)
        "0.0.0.0",
        "::1",
        "fe80::1%eth0",
        "fd00::1",          # unique local
        "::ffff:127.0.0.1", # IPv4-mapped loopback
    ],
)
def test_non_public_addresses_are_refused(resolves_to, address):
    resolves_to(address)

    with pytest.raises(BlockedURLError, match="ALLOW_PRIVATE_URLS"):
        check_url("http://internal.example/")


def test_one_private_address_among_public_ones_is_enough_to_refuse(resolves_to):
    resolves_to("93.184.215.14", "10.0.0.5")

    with pytest.raises(BlockedURLError):
        check_url("http://mixed.example/")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://example.com/", "http:///no-host"])
def test_non_http_urls_are_refused(url):
    with pytest.raises(BlockedURLError):
        check_url(url)


def test_unresolvable_host_is_a_connection_error(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", fail)

    with pytest.raises(ConnectionError):
        check_url("http://nowhere.invalid/")
