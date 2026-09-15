Test Image
     │
Reference Image
     │
     ▼
Alignment
     │
     ▼
Intensity Normalize
     │
     ▼
Difference Image
     │
     ▼
┌──────────────────────┐
│   Noise Estimation   │
│                      │
│ σ_global             │
│ σ_local(x,y)         │
│ σ_pattern(x,y)       │
└──────────┬───────────┘
           │
           ▼
Adaptive Threshold
           │
           ▼
Defect Candidate
           │
           ▼
Blob Analysis


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