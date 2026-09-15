              Input
                │
        ┌───────┴────────┐
        ▼                ▼
 Histogram           Alignment
 Threshold               │
        │                ▼
        │             Normalize
        │                │
        └───────┬────────┘
                ▼
             Difference
                │
                ▼
        Noise Estimation
                │
        ┌───────┴────────┐
        ▼                ▼
   Global Noise      Local Noise
        │                │
        └───────┬────────┘
                ▼
       Adaptive Threshold
                │
                ▼
          Defect Map
                │
                ▼
          Blob Analysis