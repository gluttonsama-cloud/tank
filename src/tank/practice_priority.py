"""Cross-process arbitration for the workbench host's official practice client."""

from contextlib import ExitStack, contextmanager
import time

from .eval_store import shard_lock


def _locks(root, team_id):
    return root / ".local/online/practice-locks" / f"team-{team_id}"


@contextmanager
def _queue(folder):
    while True:
        with shard_lock(folder, "queue") as acquired:
            if acquired:
                yield
                return
        time.sleep(0.02)


def _ticket_lock(folder, ticket):
    return shard_lock(folder, ticket.stem)


def _remove_ticket(folder, ticket):
    ticket.unlink()
    (folder / f"shard-{ticket.stem}.lock").unlink()


def _first_ticket(folder):
    with _queue(folder):
        for ticket in sorted(folder.glob("*.ticket")):
            with _ticket_lock(folder, ticket) as abandoned:
                if not abandoned:
                    return ticket
            # The OS releases a ticket lock when its owner exits unexpectedly.
            _remove_ticket(folder, ticket)
    return None


def foreground_pending(root, team_id):
    folder = _locks(root, team_id)
    if any(folder.glob("*.ticket")) and _first_ticket(folder) is not None:
        return True
    with shard_lock(folder, "host-queue") as available:
        return not available


@contextmanager
def queue_pending(root, team_id):
    """Publish queued foreground work before the host starts its next practice call."""
    deadline = time.monotonic() + 2
    while True:
        with shard_lock(_locks(root, team_id), "host-queue") as acquired:
            if acquired:
                yield
                return
        if time.monotonic() >= deadline:
            raise ValueError("主机已登记前台练习队列")
        time.sleep(0.02)


@contextmanager
def foreground(root, team_id):
    folder = _locks(root, team_id)
    with ExitStack() as lease:
        with _queue(folder):
            number = max((int(path.stem) for path in folder.glob("*.ticket")), default=0) + 1
            ticket = folder / f"{number:012d}.ticket"
            if not lease.enter_context(_ticket_lock(folder, ticket)):
                raise RuntimeError("无法登记本机官网练习")
            ticket.touch(exist_ok=False)
        try:
            while _first_ticket(folder) != ticket:
                time.sleep(0.1)
            # Keep the queue entry while the collector closes its room and socket.
            deadline = time.monotonic() + 55
            while True:
                with shard_lock(folder, "session") as acquired:
                    if acquired:
                        yield
                        return
                if time.monotonic() >= deadline:
                    raise ValueError("等待后台退出官网练习超时")
                time.sleep(0.1)
        finally:
            with _queue(folder):
                lease.close()
                _remove_ticket(folder, ticket)


def collection(root, team_id):
    return shard_lock(_locks(root, team_id), "session")
