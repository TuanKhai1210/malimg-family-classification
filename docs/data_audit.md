# MalImg audit summary — 29 September 2026

This note summarizes a prior complete read of the selected public ZIP. The original audit manifest has 9,339 rows. All 9,339 images decoded as grayscale (`L`), with no unreadable images or cross-label exact pixel duplicates. There were 8,019 unique decoded pixel groups across 25 families.

`Yuner.A` has 800 files but only one unique decoded pixel image. Keeping those files in multiple partitions would repeat the same visual input across sets; putting the entire group in one partition would leave no distinct example for held-out evaluation. The **internal plan** therefore excludes `Yuner.A` from the primary benchmark, leaving 8,539 files, 24 families and 8,018 unique pixel groups. Instructor approval remains pending.

The archive has 1,174,609,734 bytes and SHA-256 `9766ae9f1daa520e367fb486ca94728fe1485c0f5cb8314c312d77089a1fe9ec`. This hash identifies the copy audited by the group, not an author-published checksum. Exact pixel duplication is defined by hashing image mode, width, height and decoded pixel bytes. This does not detect near-duplicates or establish that malware binaries are statistically independent.

The committed code can reproduce the inventory locally. The project must still create and review its versioned 24-class split; no train, validation or test counts have been approved yet. The original 25-class manifest should not be used directly as a split.
