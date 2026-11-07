"""
Garde-fou contre la SSRF : une URL reçue d'un client ne doit pas faire
lire au serveur une ressource interne (localhost, réseau privé, endpoint
de métadonnées cloud 169.254.169.254...) dont le contenu serait ensuite
indexé puis restitué dans les réponses.

Limite connue : l'hôte est résolu ici puis de nouveau par `requests` ; un
DNS malveillant pourrait répondre différemment entre les deux (DNS
rebinding). Le contrôle de chaque redirection ferme le cas le plus courant.
"""

import ipaddress
import socket
from urllib.parse import urlsplit


class BlockedURLError(ValueError):
    """URL refusée : schéma non http(s), ou hôte résolu vers une adresse non publique."""


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # retire la zone IPv6 (fe80::1%eth0)
    # ::ffff:127.0.0.1 doit être jugée comme 127.0.0.1
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def check_url(url: str) -> None:
    """
    Lève `BlockedURLError` si `url` n'est pas en http(s) ou si l'une des
    adresses de son hôte n'est pas publique
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURLError(f"Refusing {url!r}: only http(s) URLs with a host are allowed")

    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ConnectionError(f"Impossible de résoudre {parts.hostname} : {e}") from e

    for *_, sockaddr in infos:
        if not _is_public(str(sockaddr[0])):
            raise BlockedURLError(
                f"Refusing {url!r}: {parts.hostname} resolves to the non-public address {sockaddr[0]} "
                "(set ALLOW_PRIVATE_URLS=true to ingest pages from your own network)"
            )
