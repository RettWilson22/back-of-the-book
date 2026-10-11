import threading

from backofthebook.limits import CallLimiter, Quota


def test_call_limiter_allows_a_number_of_calls_per_window():
    now = [0.0]
    limiter = CallLimiter(2, window=60, clock=lambda: now[0])

    assert limiter.try_acquire() and limiter.try_acquire()
    assert not limiter.try_acquire()
    now[0] = 59.9
    assert not limiter.try_acquire()
    now[0] = 60.0
    assert limiter.try_acquire()


def test_call_limiter_counts_each_key_separately():
    limiter = CallLimiter(1)

    assert limiter.try_acquire("203.0.113.7")
    assert not limiter.try_acquire("203.0.113.7")
    assert limiter.try_acquire("198.51.100.2")


def test_call_limiter_forgets_keys_with_no_recent_calls():
    now = [0.0]
    limiter = CallLimiter(1, window=3600, clock=lambda: now[0])
    for n in range(1000):
        limiter.try_acquire(f"visitor-{n}")

    now[0] = 3600.0
    assert limiter.try_acquire("visitor-0")
    assert list(limiter._calls) == ["visitor-0"]


def test_call_limiter_never_lets_concurrent_callers_past_the_limit():
    limiter = CallLimiter(50)
    granted: list[bool] = []

    def worker() -> None:
        for _ in range(25):
            granted.append(limiter.try_acquire("same visitor"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert granted.count(True) == 50


def test_quota_refuses_what_would_go_over_the_total_until_it_is_given_back():
    quota = Quota(100)

    assert quota.try_take(60)
    assert not quota.try_take(41)
    assert quota.try_take(40)
    quota.give_back(60)
    assert quota.try_take(60)
    assert quota.used == 100


def test_quota_never_lets_concurrent_callers_take_more_than_the_total():
    quota = Quota(500)
    taken: list[bool] = []

    def worker() -> None:
        for _ in range(100):
            taken.append(quota.try_take(3))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert taken.count(True) == 500 // 3
    assert quota.used == 3 * (500 // 3)
