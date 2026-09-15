Raw Image
    |
    v
Pre Processing
    |
    |
    +----------------+
    |                |
Flat Field       Noise Filter
Correction
    |
    v

Die Alignment
(Affine/ECC/Phase Correlation)

    |
    v

Intensity Normalize

    |
    v


Pixel Difference


Diff =
abs(TestDie - RefDie)


    |
    v

Noise Estimation

    |
    v

Adaptive Threshold

    |
    v

Binary Defect Map

    |
    v

Blob Analysis

    |
    v

Defect Classification

    |
    v

Defect Database