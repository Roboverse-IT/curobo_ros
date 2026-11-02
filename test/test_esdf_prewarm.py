from curobo_ros.core.unified_planner_node import _EsdfPrewarm


def test_prewarm_waits_for_a_successful_integration_then_runs_once():
    warmup = _EsdfPrewarm(True)

    assert warmup.snapshot()[0] == 'waiting_for_integration'
    assert warmup.request_after_integration()
    assert warmup.take_request()
    warmup.finish(True)

    assert warmup.snapshot()[0] == 'ready'
    assert not warmup.request_after_integration()
    assert not warmup.take_request()


def test_prewarm_failure_retries_a_bounded_number_of_times():
    warmup = _EsdfPrewarm(True, max_attempts=3)
    assert warmup.request_after_integration()

    for attempt in range(2):
        assert warmup.take_request()
        warmup.finish(False, f'failure {attempt}')
        assert warmup.snapshot()[0] == 'requested'

    assert warmup.take_request()
    warmup.finish(False, 'final failure')

    state, attempts, error = warmup.snapshot()
    assert (state, attempts, error) == ('failed', 3, 'final failure')
    assert not warmup.request_after_integration()


def test_disabled_prewarm_never_requests_cuda_work():
    warmup = _EsdfPrewarm(False)
    assert warmup.snapshot()[0] == 'disabled'
    assert not warmup.request_after_integration()
    assert not warmup.take_request()
