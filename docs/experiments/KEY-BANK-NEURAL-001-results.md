# KEY-BANK-NEURAL-001: resource failure, accuracy unavailable

The single registered attempt exceeded its 2 GiB sampled MPS driver allocation
limit before producing any complete case/model prediction. This does not test
whether the neural sources improve the candidate ranking. No run was restarted,
no limit was raised, and no completed case was omitted.

Source `d5d4ea38118a21c83260950c2f8a749c8bd6187a` was published and remote-verified
before prediction. Terminal failure and candidate inventories were published at
`d9b405d8475b9a38130648dc8e2d71603fcb0ac6` before the one evaluation. Both seeds
receive the predeclared 3,584/3,584 missing-output penalty, with zero supported
records and a failed development gate. **That penalty is not an observed 100%
neural error rate:** no complete reading was available to measure.

All eight candidate inventories were prepared: 104,290 tuples and 79,264 distinct
records counted separately per case. All eight original statistical winners and
scores were reproduced. Some neural batch computation began, but incomplete
batch scores were not retained. The 16 scheduled case/model results remain absent.

Measured prediction work: 27.270784 wall seconds, 26.974039 host CPU seconds,
699,088,896 bytes peak host RSS, zero paid spend. Host RSS below 4 GiB identifies
the driver-memory branch of the guard; the exact GPU driver peak was not saved.
Evaluation took 0.001409 measured wall seconds and independently checked 32
missing-record edit penalties. Entry admission/hash checks are outside these
phase timers. Artifact accounting binds all eight prepared archives and the
failure record; compact evidence is under `results/KEY-BANK-NEURAL-001/`.

The original statistical 43/3,584 result and the fresh CONFIRM-002 protocol remain
unchanged. A separate artificial-input memory benchmark can qualify an engineering
revision, which must receive a new experiment ID and retain this failure.
