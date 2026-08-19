"""eoio.utils.tests.test_aux_read - tests for eoio.utils.aux_read"""

import unittest

from eoio.utils.aux_read import warn_on_aux_failure


class TestWarnOnAuxFailure(unittest.TestCase):
    def test_no_exception_runs_block_normally(self):
        result = []
        with warn_on_aux_failure("some aux source"):
            result.append(1)

        self.assertEqual(result, [1])

    def test_exception_is_caught_and_warned(self):
        with self.assertWarns(UserWarning) as ctx:
            with warn_on_aux_failure("some aux source"):
                raise ValueError("boom")

        self.assertIn("some aux source", str(ctx.warning))
        self.assertIn("boom", str(ctx.warning))

    def test_exception_does_not_propagate(self):
        # Should not raise -- that's the whole point.
        with warn_on_aux_failure("some aux source"):
            raise RuntimeError("this should be swallowed")

    def test_partial_work_before_exception_is_kept(self):
        """Code inside the block runs up to the failure point -- callers relying on
        e.g. a list.append() before the failing line still see that partial effect,
        since the context manager only stops propagation, it doesn't roll anything back."""
        parts = []
        with warn_on_aux_failure("some aux source"):
            parts.append("first")
            raise ValueError("second part failed")

        self.assertEqual(parts, ["first"])


if __name__ == "__main__":
    unittest.main()
