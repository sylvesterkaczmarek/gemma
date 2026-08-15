# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Hackable Diffusion Gemma network wrapper."""

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
import jax.numpy as jnp
import numpy as np


class _NonZeroZeroLogitEmbedder:
  """Embedder whose zero logits deliberately map to a nonzero vector."""

  def encode_logits(self, logits):
    return jnp.ones(logits.shape[:-1] + (3,), dtype=jnp.float32)


class SelfConditioningEncodingTest(absltest.TestCase):

  def test_missing_self_conditioning_is_exact_zero_after_encoding(self):
    embedder = _NonZeroZeroLogitEmbedder()

    raw_zero_encoding = embedder.encode_logits(jnp.zeros((2, 4, 5)))
    self.assertTrue(bool(jnp.all(raw_zero_encoding != 0)))

    result = hd_gemma_network._encode_self_conditioning(
        embedder,
        sc_logits=None,
        sc_mask=None,
        batch_size=2,
        canvas_length=4,
        vocab_size=5,
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(result, jnp.zeros_like(result))

  def test_zero_logit_sentinel_disables_only_matching_examples(self):
    embedder = _NonZeroZeroLogitEmbedder()
    sc_logits = jnp.stack([
        jnp.ones((4, 5), dtype=jnp.float32),
        jnp.zeros((4, 5), dtype=jnp.float32),
    ])

    result = hd_gemma_network._encode_self_conditioning(
        embedder,
        sc_logits=sc_logits,
        sc_mask=None,
        batch_size=2,
        canvas_length=4,
        vocab_size=5,
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(result[0], jnp.ones_like(result[0]))
    np.testing.assert_array_equal(result[1], jnp.zeros_like(result[1]))

  def test_explicit_mask_overrides_zero_logit_sentinel_inference(self):
    embedder = _NonZeroZeroLogitEmbedder()
    sc_logits = jnp.ones((2, 4, 5), dtype=jnp.float32)
    sc_mask = jnp.array([True, False])

    result = hd_gemma_network._encode_self_conditioning(
        embedder,
        sc_logits=sc_logits,
        sc_mask=sc_mask,
        batch_size=2,
        canvas_length=4,
        vocab_size=5,
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(result[0], jnp.ones_like(result[0]))
    np.testing.assert_array_equal(result[1], jnp.zeros_like(result[1]))


if __name__ == '__main__':
  absltest.main()
