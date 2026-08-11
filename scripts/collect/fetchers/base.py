"""Fetcher 인터페이스."""

from typing import Protocol, runtime_checkable

from ..models import RawDoc, Source


class LicensePendingError(RuntimeError):
    """license: pending-check 인 소스를 수집하려 할 때.

    robots.txt/ToS 확인 전에는 수집하지 않는다는 정책을 코드로 강제한다.
    """


class RobotsDisallowedError(RuntimeError):
    """robots.txt가 수집을 거부한 URL.

    조용히 건너뛰면 '왜 이 문서가 코퍼스에 없지?'를 나중에 알 수 없으므로
    스킵하지 않고 에러를 낸다.
    """


@runtime_checkable
class Fetcher(Protocol):
    async def fetch(self, source: Source) -> list[RawDoc]:
        """소스에서 문서를 수집한다. URL 하나당 RawDoc 하나."""
        ...


def ensure_license_checked(source: Source) -> None:
    if source.license_pending:
        raise LicensePendingError(
            f"소스 {source.id!r}는 license: pending-check 상태입니다. "
            "robots.txt/이용약관을 확인하고 sources.yaml의 license를 갱신한 뒤 수집하세요."
        )
