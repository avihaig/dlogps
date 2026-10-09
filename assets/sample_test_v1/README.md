# sample_test_v1

The same two cables as `../sample_v1/`, released from poses `sample_v1` never
used (`configs/release_test.yaml`: only the seed differs). They come from the
second training half: episodes 50 and 51 of each cable in `release_train`. Two episodes
each (829, 698 frames for cable 000; 1199, 1211 for cable 045). `scripts/smoke.sh`
rolls out on these after training on `sample_v1`. Format: docs/data.md.
