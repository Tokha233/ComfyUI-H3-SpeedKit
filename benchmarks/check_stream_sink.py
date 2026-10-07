#!/usr/bin/env python3
"""Opt-in CUDA checks for RGB transfer ownership and callback failure cleanup."""
import argparse
from pathlib import Path
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfyui", type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.comfyui.resolve())]
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ["sink-check", "--disable-cuda-malloc"]
    import main as comfy_main
    import torch
    from h3_speedkit.video import RGBSink

    def check(counts):
        torch.cuda.set_device(0)
        with torch.cuda.stream(torch.cuda.Stream()), torch.inference_mode():
            expected = torch.arange(sum(counts) * 16 * 24 * 3, dtype=torch.int32).byte().view(-1, 16, 24, 3)
            position = 0
            seen = []

            def consume(start, rgb):
                # Slow consumers force producer backpressure and repeated slot reuse.
                time.sleep(.003)
                seen.append((start, rgb.clone()))

            sink = RGBSink((1, 3, sum(counts), 16, 24), torch.device("cuda:0"), consume)
            try:
                for count in counts:
                    tensor = expected[position:position + count].cuda()
                    sink.write(tensor)
                    del tensor
                    # Stress allocator reuse while the D2H copy is in flight.
                    torch.empty((count, 16, 24, 3), device="cuda", dtype=torch.uint8).fill_(17)
                    position += count
                output = sink.finish()
                assert torch.equal(output, expected)
                assert torch.equal(torch.cat([rgb for _, rgb in seen]), expected)
                offset = 0
                for start, rgb in seen:
                    assert start == offset
                    offset += len(rgb)
            finally:
                sink.close()

    with torch.inference_mode():
        for counts in ([1], [3, 1], [1, 7, 2], [7, 2, 5, 1], [2] * 17):
            check(counts)
        errors = []

        def concurrent():
            try:
                check([1, 7, 2, 5, 3])
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=concurrent) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
            assert not thread.is_alive()
        assert not errors, errors

        def fail(start, rgb):
            raise OSError("consumer failed")

        sink = RGBSink((1, 3, 2, 16, 24), torch.device("cuda:0"), fail)
        try:
            sink.write(torch.zeros((2, 16, 24, 3), device="cuda", dtype=torch.uint8))
            try:
                sink.finish()
            except OSError as error:
                assert str(error) == "consumer failed"
            else:
                raise AssertionError("Lost callback failure")
        finally:
            sink.close()
        assert not any(t.name.startswith("h3-rgb-drain") for t in threading.enumerate())
    print("PASS: 5 chunk layouts, 2 concurrent sinks, callback failure, thread cleanup")


if __name__ == "__main__":
    main()
