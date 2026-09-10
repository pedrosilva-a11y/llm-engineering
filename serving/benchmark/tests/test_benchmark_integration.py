"""Integration tests for the LLM serving benchmark harness."""

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

from serving.benchmark.client import BenchmarkClient, BenchmarkClientConfiguration
from serving.benchmark.stub_server import StubServerConfiguration, create_stub_app
from serving.benchmark.workload import WorkloadRequest


@pytest.fixture
def anyio_backend() -> str:
    """Run async integration tests with the asyncio backend."""
    return "asyncio"


@pytest.fixture
def stub_server_url() -> Iterator[str]:
    """Run the benchmark stub through a real local HTTP server."""
    application = create_stub_app(
        StubServerConfiguration(
            first_token_delay_seconds=0.1,
            inter_token_delay_seconds=0.03,
            token_id=7,
        )
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen()

    port = server_socket.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            application,
            log_level="warning",
            lifespan="off",
        )
    )

    server_thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [server_socket]},
        daemon=True,
    )
    server_thread.start()

    try:
        deadline = time.monotonic() + 5.0

        while not server.started:
            if not server_thread.is_alive():
                raise RuntimeError("stub server stopped during startup.")

            if time.monotonic() >= deadline:
                raise RuntimeError("stub server did not start within timeout.")

            time.sleep(0.01)

        yield f"http://127.0.0.1:{port}"

    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)
        server_socket.close()


@pytest.mark.anyio
async def test_benchmark_client_measures_known_stub_delays(
    stub_server_url: str,
) -> None:
    """Measure known TTFT and ITL delays through real streaming HTTP."""
    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url=f"{stub_server_url}/v1/completions",
            model_name="stub-model",
        )
    )

    execution = await client.run_fixed_workload(
        (
            WorkloadRequest(
                request_id="request-000000",
                prompt_token_ids=(1, 2, 3),
                max_tokens=4,
                arrival_offset_seconds=0.0,
            ),
        )
    )

    metrics = execution.request_metrics[0]

    assert metrics.output_tokens == 4

    assert metrics.ttft_ms == pytest.approx(100.0, abs=20.0)

    assert metrics.itl_ms == pytest.approx((30.0, 30.0, 30.0), abs=20.0)

    assert metrics.tpot_ms == pytest.approx(30.0, abs=20.0)

    assert metrics.e2e_ms == pytest.approx(190.0, abs=30.0)
