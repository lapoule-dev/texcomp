// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! How hard the kernels search, as numbers.

/// How hard the encoder searches for each block's encoding: numbers, each a
/// trade of GPU time for quality, compiled into the kernel by
/// [`GpuEncoder::with_quality`](super::GpuEncoder::with_quality).
///
/// [`Quality::level`] derives all four from one number, 0 (fastest) to
/// [`Quality::MAX_LEVEL`]; [`Quality::DEFAULT`] is the level the measured
/// table in the README picks. A setting can also be written field by field.
/// Values beyond a field's bounds are clamped to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Quality {
    /// The encodings tried per block, 0 to 2. Each tier adds to the one
    /// below it:
    ///
    /// | Tier | BC7, opaque block | BC7, block with alpha | ASTC 4×4, opaque | ASTC 4×4, with alpha |
    /// |---|---|---|---|---|
    /// | 0 | modes 6, 1 | mode 6 | 1 partition, RGB direct, weights 0..11 | 1 partition, RGBA direct, weights 0..5 |
    /// | 1 | + modes 3, 5 | + modes 5, 7 | + weight ranges 0..7 to 0..31, luminance, RGB base + scale | + weights 0..7, 0..11 |
    /// | 2 | + modes 0, 2, 4 | + mode 4 | + 2 partitions, RGB direct, weights 0..2 to 0..7 | + 2 partitions, RGBA direct |
    pub mode_tier: u32,
    /// Partitions fully fitted per partitioned encoding, 1 to 16: the best
    /// by a cheap estimate of each one (the variance it leaves off a line
    /// per subset).
    pub partition_candidates: u32,
    /// Endpoint fit iterations, 1 to 8: quantise the endpoints, give each
    /// texel its nearest weight, refit the endpoints to those weights by
    /// least squares.
    pub refine_iterations: u32,
    /// How much the error each choice minimises weighs luma, in percent, 0
    /// to 100: 0 is the squared error summed over R, G and B; 100 weighs the
    /// channels by their part in BT.601 luma (0.299, 0.587, 0.114), which
    /// keeps the eye's main signal, at a cost to the error of blue.
    pub luma_weighting: u32,
}

impl Quality {
    /// The highest [`Quality::level`].
    pub const MAX_LEVEL: u32 = 4;

    /// The level the encoder uses unless told otherwise: see the README's
    /// table, from which it was chosen.
    pub const DEFAULT: Quality = Quality::level(1);

    /// The setting of level `n`, 0 to [`Quality::MAX_LEVEL`] (beyond it,
    /// the highest):
    ///
    /// | Level | `mode_tier` | `partition_candidates` | `refine_iterations` | `luma_weighting` |
    /// |---|---|---|---|---|
    /// | 0 | 0 | 1 | 3 | 0 |
    /// | 1 | 1 | 2 | 3 | 50 |
    /// | 2 | 2 | 3 | 3 | 50 |
    /// | 3 | 2 | 6 | 4 | 50 |
    /// | 4 | 2 | 12 | 6 | 50 |
    ///
    /// Level 0 is the search of texcomp 0.3: BC7 modes 6 and 1 with one
    /// partition candidate, ASTC with one configuration per block kind, the
    /// plain RGB error.
    pub const fn level(n: u32) -> Quality {
        let (mode_tier, partition_candidates, refine_iterations, luma_weighting) = match n {
            0 => (0, 1, 3, 0),
            1 => (1, 2, 3, 50),
            2 => (2, 3, 3, 50),
            3 => (2, 6, 4, 50),
            _ => (2, 12, 6, 50),
        };
        Quality {
            mode_tier,
            partition_candidates,
            refine_iterations,
            luma_weighting,
        }
    }

    /// The setting with every field within its bounds.
    pub const fn clamped(self) -> Quality {
        const fn clamp(v: u32, lo: u32, hi: u32) -> u32 {
            if v < lo {
                lo
            } else if v > hi {
                hi
            } else {
                v
            }
        }
        Quality {
            mode_tier: clamp(self.mode_tier, 0, 2),
            partition_candidates: clamp(self.partition_candidates, 1, 16),
            refine_iterations: clamp(self.refine_iterations, 1, 8),
            luma_weighting: clamp(self.luma_weighting, 0, 100),
        }
    }

    /// The error weight of R, G, B: 3 in total, blended from equal weights
    /// towards luma's by `luma_weighting`.
    pub(crate) fn channel_weights(self) -> [f32; 3] {
        let t = self.clamped().luma_weighting as f32 / 100.0;
        let luma = [0.299f32, 0.587, 0.114];
        luma.map(|l| (1.0 - t) + t * 3.0 * l)
    }
}

impl Default for Quality {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every level is within bounds, and they climb**: no field falls from
    /// one level to the next.
    #[test]
    fn levels_are_bounded_and_monotonic() {
        let mut previous = Quality::level(0);
        for n in 0..=Quality::MAX_LEVEL + 2 {
            let q = Quality::level(n);
            assert_eq!(q, q.clamped(), "level {n}");
            assert!(q.mode_tier >= previous.mode_tier);
            assert!(q.partition_candidates >= previous.partition_candidates);
            assert!(q.refine_iterations >= previous.refine_iterations);
            assert!(q.luma_weighting >= previous.luma_weighting);
            previous = q;
        }
        assert_eq!(Quality::level(99), Quality::level(Quality::MAX_LEVEL));
        assert_eq!(Quality::default(), Quality::DEFAULT);
    }

    /// **Out-of-bounds fields are clamped**, and the channel weights always
    /// total 3.
    #[test]
    fn fields_clamp_and_weights_total_three() {
        let q = Quality {
            mode_tier: 9,
            partition_candidates: 0,
            refine_iterations: 100,
            luma_weighting: 400,
        }
        .clamped();
        assert_eq!((q.mode_tier, q.partition_candidates), (2, 1));
        assert_eq!((q.refine_iterations, q.luma_weighting), (8, 100));
        for luma in [0, 50, 100] {
            let w = Quality {
                luma_weighting: luma,
                ..Quality::DEFAULT
            }
            .channel_weights();
            assert!((w.iter().sum::<f32>() - 3.0).abs() < 1e-5, "{w:?}");
        }
        assert_eq!(Quality::level(0).channel_weights(), [1.0, 1.0, 1.0]);
    }
}
